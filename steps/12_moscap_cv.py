"""12. MOS capacitor vs hand calculations: flat-band and threshold voltage, interface charge shift
-qQ/Cox, quasi-static C-V, at 300 and 423 K.

  python steps/12_moscap_cv.py

Hand calculation (p-type, ionized holes p0 from incomplete ionization):
  psi_B = -Vt ln(p0/ni)            (bulk potential, intrinsic-level reference)
  Vfb   = (phi_m - phi_i) + psi_B - qQ/Cox
  Vth   = Vfb + 2|psi_B| + sqrt(2 eps q NA 2|psi_B|)/Cox     (depletion charge uses the full NA)
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from, gate_from
from sicdev import devices as dv, physics as ph, solver
from sicdev.params import Q, EPS0, gate_offset
from sicdev.utils import reset_devsim
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "12"
cfg = load_config()
mat, gate = material_from(cfg), gate_from(cfg)
g = cfg["moscap"]
COX = 3.9 * EPS0 / (g["tox_um"] * 1e-4)
print(f"Cox = {COX:.4e} F/cm^2, q/Cox = {Q / COX * 1e12:.4f} V per 1e12 cm^-2")


def analytic(T, qeff):
    d = mat.derived(T)
    NA = g["NA"]
    a = mat.gA / d["p1A"]
    p0 = (-1 + math.sqrt(1 + 4 * a * NA)) / (2 * a)
    psib = -d["Vt"] * math.log(p0 / d["ni"])
    vfb = gate_offset(mat, gate, T) + psib - Q * qeff / COX
    phi2 = 2 * abs(psib)
    qd = math.sqrt(2 * mat.eps_r * EPS0 * Q * NA * phi2)
    return dict(psib=psib, vfb=vfb, vth=vfb + phi2 + qd / COX)


def run(T, qeff, dd=False, vmax=15.0, ii=True):
    """dd=False: equilibrium Poisson (DC state of a MOS capacitor: no current, flat Fermi level).
    dd=True : full drift-diffusion, only for accumulation/depletion as a consistency check."""
    reset_devsim()
    ph.set_extended_precision(True)
    t0 = time.time()
    with solver.quiet():
        lay = dv.build_moscap("mc", cfg)
        dv.setup_physics("mc", lay, mat, T, ii=ii, interface=dict(qf_cm2=0.0, qit_eff_cm2=qeff), gate=gate)
    tol = cfg["numerics"]["tol"] if dd else dv.POISSON_TOL
    if dd:
        with solver.quiet():
            dv.equilibrate("mc", lay, ii=ii, tol=tol, holes=True)
    else:
        solver.solve_dc("mc", dv.POISSON_TOL)
    data = []

    def rec(v):
        psi = ph.node_array("mc", "sic", "Potential")
        data.append((v, psi[0], psi[-1], ph.contact_charge("mc", "gate")))

    rec(0.0)
    solver.ramp("mc", "gate", -10.0, 0.25, callback=rec, tol=tol)
    solver.ramp("mc", "gate", 0.0, 1.0, tol=tol)
    solver.ramp("mc", "gate", vmax, 0.25, callback=rec, tol=tol)
    vg, psis, psib, qg = np.array(sorted(data)).T
    return dict(T=T, qeff=qeff, vg=vg, psis=psis, psib=psib[0], c=np.gradient(qg, vg) / COX,
                vfb=float(np.interp(0.0, psis - psib, vg)),      # psi_s = psi_bulk
                vth=float(np.interp(0.0, psis + psib, vg)),      # psi_s = -psi_bulk (n_s = p0)
                t=time.time() - t0)


res = [run(T, q) for T in (300.0, 423.0) for q in (0.0, 1e12, -1e12)]
print(f"\n{'T':>4} {'Q (cm-2)':>9} {'Vfb sim':>8} {'Vfb hand':>8} {'Vth sim':>8} {'Vth hand':>8} {'psiB sim':>8}")
for r in res:
    a = analytic(r["T"], r["qeff"])
    print(f"{r['T']:4.0f} {r['qeff']:+9.0e} {r['vfb']:+8.3f} {a['vfb']:+8.3f} {r['vth']:+8.3f} {a['vth']:+8.3f} "
          f"{r['psib']:8.4f}")
print("\nflat-band shift by the interface charge:")
for T in (300.0, 423.0):
    base = [r for r in res if r["T"] == T and r["qeff"] == 0][0]
    for r in res:
        if r["T"] == T and r["qeff"] != 0:
            print(f"  {T:.0f} K, Q = {r['qeff']:+.0e}: dVfb sim = {r['vfb'] - base['vfb']:+.4f} V, "
                  f"-qQ/Cox = {-Q * r['qeff'] / COX:+.4f} V")
ddc = run(300.0, 0.0, dd=True, vmax=1.0)
po = res[0]
diff = max(abs(np.interp(v, ddc["vg"], ddc["psis"]) - np.interp(v, po["vg"], po["psis"])) for v in ddc["vg"])
print(f"\nfull drift-diffusion vs equilibrium Poisson, VG -10..+1 V: max |psi_s difference| = {diff:.1e} V")
off = run(300.0, 0.0, ii=False)
print("\nC/Cox near flat band, 300 K, Q = 0 (incomplete ionization on / off):")
for v in (-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0):
    print(f"  VG = {v:+.2f} V: {np.interp(v, po['vg'], po['c']):.4f} / {np.interp(v, off['vg'], off['c']):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for r in res:
    ls = {0.0: "-", 1e12: "--", -1e12: ":"}[r["qeff"]]
    col = "tab:blue" if r["T"] == 300 else "tab:red"
    lab = f"{r['T']:.0f} K, Q = {r['qeff']:+.0e}"
    ax[0].plot(r["vg"], r["psis"] - r["psib"], ls, color=col, label=lab)
    ax[1].plot(r["vg"], r["c"], ls, color=col, label=lab)
ax[0].set(xlabel="V$_G$ (V)", ylabel="$\\psi_s - \\psi_{bulk}$ (V)", title="surface band bending")
ax[1].set(xlabel="V$_G$ (V)", ylabel="C / C$_{ox}$", ylim=(0, 1.05), title="quasi-static C-V")
ax[0].legend(fontsize=7)
ax[1].legend(fontsize=7, loc="lower left")
savefig(fig, IMG / "moscap_cv.png")
