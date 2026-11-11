"""11. 1D MOS capacitor (n+ poly / 50 nm SiO2 / p-type SiC 1e17): gate boundary condition, oxide,
interface, and band diagrams from accumulation to inversion (equilibrium Poisson).

  python steps/11_moscap_bands.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from, gate_from
from sicdev import devices as dv, physics as ph, solver
from sicdev.params import gate_offset
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "11"
cfg = load_config()
mat, gate = material_from(cfg), gate_from(cfg)
T = 300.0
d = mat.derived(T)
print(f"gate work function {gate.phi_m} eV, SiC: chi = {mat.chi} eV, Ec - Ei = {d['Ec_minus_Ei']:.4f} eV")
print(f"gate boundary condition: Potential = VG - (phi_m - chi - (Ec - Ei)) = VG - ({gate_offset(mat, gate, T):+.4f} V)")

ph.set_extended_precision(True)
with solver.quiet():
    lay = dv.build_moscap("mc", cfg)
    dv.setup_physics("mc", lay, mat, T, ii=True, interface=dict(qf_cm2=0.0, qit_eff_cm2=0.0), gate=gate)
solver.solve_dc("mc", dv.POISSON_TOL)
xs = ph.node_array("mc", "sic", "x") * 1e4
xo = ph.node_array("mc", "oxide", "x") * 1e4
print(f"nodes: oxide {len(xo)}, SiC {len(xs)}")

profiles = {}
targets = (-6.0, -2.65, 1.0, 8.0)       # accumulation, ~flat band, depletion, inversion


def grab(v):
    if any(abs(v - t) < 1e-9 for t in targets):
        profiles[round(v, 2)] = (ph.node_array("mc", "oxide", "Potential"), ph.node_array("mc", "sic", "Potential"))


grab(0.0)
solver.ramp("mc", "gate", -6.0, 0.05, tol=dv.POISSON_TOL, callback=grab)
for t in (-2.65, 1.0, 8.0):
    solver.ramp("mc", "gate", t, 0.25, tol=dv.POISSON_TOL)
    grab(t)

print(f"{'VG (V)':>7} {'psi_s - psi_bulk (V)':>21} {'n surface (cm-3)':>17} {'p surface (cm-3)':>17}")
for v, (po, ps) in sorted(profiles.items()):
    n_s, p_s = d["ni"] * np.exp(ps[0] / d["Vt"]), d["ni"] * np.exp(-ps[0] / d["Vt"])
    print(f"{v:7.2f} {ps[0] - ps[-1]:21.4f} {n_s:17.3e} {p_s:17.3e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
names = {-6.0: "accumulation", -2.65: "flat band", 1.0: "depletion", 8.0: "inversion"}
for v, (po, ps) in sorted(profiles.items()):
    ax[0].plot(np.r_[xo, xs] * 1e3, np.r_[po, ps], label=f"VG = {v:+.2f} V ({names[v]})")
    Ec = -ps + d["Ec_minus_Ei"]
    l, = ax[1].plot(xs * 1e3, Ec, label=f"{names[v]}")
    ax[1].plot(xs * 1e3, Ec - d["Eg"], color=l.get_color())
ax[0].axvspan(-50, 0, color="gray", alpha=0.15)
ax[0].text(-45, ax[0].get_ylim()[1] * 0.85, "SiO$_2$", fontsize=9)
ax[0].set(xlim=(-55, 400), xlabel="depth (nm)", ylabel="Potential (V)", title="potential (intrinsic-level reference)")
ax[0].legend(fontsize=7)
ax[1].axhline(0, color="k", ls="--", lw=0.8)
ax[1].text(320, 0.1, "E$_F$", fontsize=9)
ax[1].set(xlim=(0, 400), xlabel="depth (nm)", ylabel="energy (eV)", title="SiC bands (E$_c$, E$_v$), 300 K")
ax[1].legend(fontsize=7, loc="lower right")
savefig(fig, IMG / "moscap_bands.png")
