"""10. Check the 1D PN diode against hand calculations (300 / 423 K, ionization on / off),
then the forward I-V in 128-bit and in double precision.

  python steps/10_pn_check.py
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from
from sicdev import devices as dv, physics as ph, solver
from sicdev.params import Q, EPS0
from sicdev.solver import ramp
from sicdev.utils import reset_devsim
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "10"
cfg = load_config()
mat = material_from(cfg)
g = cfg["pn_diode"]
tol = cfg["numerics"]["tol"]


def analytic(T, ii):
    """Vbi = Vt ln(p0 n0 / ni^2) with the ionized majority densities; depletion width uses the full doping."""
    d = mat.derived(T)
    NA, ND = g["NA"], g["ND"]
    if ii:
        a = mat.gA / d["p1A"]
        p0 = (-1 + math.sqrt(1 + 4 * a * NA)) / (2 * a)
        b = mat.gD / d["n1D"]
        n0 = (-1 + math.sqrt(1 + 4 * b * ND)) / (2 * b)
    else:
        p0, n0 = NA, ND
    vbi = d["Vt"] * math.log(p0 * n0 / d["ni"] ** 2)
    W = math.sqrt(2 * mat.eps_r * EPS0 * vbi / Q * (NA + ND) / (NA * ND))
    return dict(p0=p0, n0=n0, vbi=vbi, xn_um=W * NA / (NA + ND) * 1e4)


def run(T, ii, extended=True):
    reset_devsim()
    ph.set_extended_precision(extended)
    t0 = time.time()
    with solver.quiet():
        lay = dv.build_pn_diode("pn", cfg)
        dv.setup_physics("pn", lay, mat, T, ii=ii)
        dv.equilibrate("pn", lay, ii=ii, tol=tol, holes=True)
    x = ph.node_array("pn", "sic", "x") * 1e4
    psi = ph.node_array("pn", "sic", "Potential")
    n = ph.node_array("pn", "sic", "Electrons")
    p = ph.node_array("pn", "sic", "Holes")
    i_p, i_n = np.argmin(abs(x - 0.5)), np.argmin(abs(x - 8.0))
    xn = x[(x > g["xj_um"]) & (n >= 0.5 * n[i_n])].min() - g["xj_um"]   # n reaches half its neutral value
    res = dict(T=T, ii=ii, vbi=psi[-1] - psi[0], p0=p[i_p], xn_um=xn, V=[], J=[])
    ramp("pn", "anode", 3.2, 0.1, tol=tol,
         callback=lambda v: (res["V"].append(v), res["J"].append(ph.contact_current("pn", "anode"))))
    res["t"] = time.time() - t0
    return res


cases = [run(T, ii) for T in (300.0, 423.0) for ii in (True, False)]
print(f"{'T':>4} {'ion.':>5} {'Vbi sim':>9} {'Vbi hand':>9} {'p0 sim':>10} {'p0 hand':>10} {'ionized':>8} "
      f"{'xn sim':>7} {'xn hand':>7}")
for r in cases:
    a = analytic(r["T"], r["ii"])
    print(f"{r['T']:4.0f} {'on' if r['ii'] else 'off':>5} {r['vbi']:9.4f} {a['vbi']:9.4f} {r['p0']:10.3e} {a['p0']:10.3e} "
          f"{100 * r['p0'] / g['NA']:7.1f}% {r['xn_um']:7.3f} {a['xn_um']:7.3f}")
print("(xn in um; xn sim = where n reaches half of its neutral value)")

dbl = run(300.0, True, extended=False)
ext = cases[0]
print("\nforward current at 300 K, ionization on:")
print(f"{'V (V)':>6} {'J 128-bit (A/cm2)':>18} {'J double (A/cm2)':>17}")
for k in (0, 4, 9, 14, 19, 24, 27, 29, 31):
    print(f"{ext['V'][k]:6.1f} {ext['J'][k]:18.3e} {dbl['J'][k]:17.3e}")
print(f"runtime per case: {', '.join(f'{r['t']:.1f}' for r in cases)} s (128-bit), {dbl['t']:.1f} s (double)")

fig, ax = plt.subplots(figsize=(5.8, 4))
for r in cases:
    ax.semilogy(r["V"], np.abs(r["J"]), "-" if r["ii"] else "--",
                label=f"{r['T']:.0f} K, ionization {'on' if r['ii'] else 'off'}")
ax.semilogy(dbl["V"], np.abs(dbl["J"]), "k.", ms=4, label="300 K, on, double precision")
ax.set(xlabel="V$_{anode}$ (V)", ylabel="|J| (A/cm$^2$)", title="4H-SiC PN diode, forward I-V", ylim=(1e-27, 1e5))
ax.legend(loc="lower right")
savefig(fig, IMG / "pn_forward_iv.png")
