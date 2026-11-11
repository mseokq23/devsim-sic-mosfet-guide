"""18. Output curve ID-VDS at VGS = 18 V (300 K), Ron,sp, where the voltage drops, and the current flow.

  python steps/18_output.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from
from sicdev import mosfet, physics as ph, solver
from sicdev.extract import ron_sp
from sicdev.plot import plt, savefig, triangulation, element_vector

IMG = ROOT / "docs" / "img" / "18"
cfg = load_config()
b, tol = cfg["bias"], cfg["numerics"]["tol"]
t0 = time.time()
lay = mosfet.build(cfg, 300.0)
W = lay["info"]["width_cm"]
d = material_from(cfg).derived(300.0)
solver.ramp("m", "drain", b["vds_lin"], 0.05, tol=tol)
for stop, step in b["vgs_segments"]:            # same path as the protocol: 0 -> 20 V, then back to 18 V
    solver.ramp("m", "gate", stop, step, tol=tol)
solver.ramp("m", "gate", b["vgs_on"], 0.5, tol=tol)

x = ph.node_array("m", "sic", "x") * 1e4
y = ph.node_array("m", "sic", "y") * 1e4


def phi_n():
    """Electron quasi-Fermi potential: n = ni exp((Potential - phi_n)/Vt)."""
    psi, n = ph.node_array("m", "sic", "Potential"), ph.node_array("m", "sic", "Electrons")
    return psi - d["Vt"] * np.log(n / d["ni"])


def at(xq, yq):
    return int(np.argmin((x - xq) ** 2 + (y - yq) ** 2))


VD, ID = [b["vds_lin"]], [ph.contact_current("m", "drain")]
# where does VDS = 0.1 V drop? phi_n along the current path (rows averaged across x where noted)
pf = phi_n()
pts = [("source contact", 0.0), ("channel start (x=1.80, y=0)", pf[at(1.80, 0.0)]),
       ("channel end (x=2.30, y=0)", pf[at(2.30, 0.0)]), ("JFET bottom (x=3.45, y=1.0)", pf[at(3.45, 1.0)]),
       ("drift bottom (x=3.45, y=10.0)", pf[at(3.45, 10.0)]), ("drain contact", b["vds_lin"])]
print(f"(check: phi_n at y = 10 um is {pf[at(0.05, 10.0)] * 1e3:.2f} mV at x = 0.05 um and "
      f"{pf[at(3.45, 10.0)] * 1e3:.2f} mV at x = 3.45 um)")
print(f"VGS = {b['vgs_on']} V, VDS = {b['vds_lin']} V: electron quasi-Fermi potential along the path")
for (n0, v0), (n1, v1) in zip(pts[:-1], pts[1:]):
    print(f"  {n0:>30} -> {n1:<30}: {1e3 * (v1 - v0):6.1f} mV ({100 * (v1 - v0) / b['vds_lin']:4.1f} %)")


def rec(v):
    VD.append(v)
    ID.append(ph.contact_current("m", "drain"))


for stop, step in b["vds_segments"]:
    solver.ramp("m", "drain", stop, step, tol=tol, callback=rec)
VD, ID = np.array(VD), np.array(ID)
ron = ron_sp(VD, ID, cfg["extract"]["ron_vds_max"], W)
print(f"\n{len(VD)} points, {solver.STATS.n_solves} solves ({solver.STATS.n_fail} failed), {time.time() - t0:.1f} s")
print(f"{'VDS (V)':>8} {'ID (A/cm)':>10} {'JD (A/cm2)':>11}")
for v, i in zip(VD, ID):
    print(f"{v:8.2f} {i:10.4f} {i / W:11.2f}")
print(f"Ron,sp (VDS <= {cfg['extract']['ron_vds_max']} V, line through the origin) = {ron:.4f} mOhm cm^2")

tri = triangulation("m", "sic")
jx, jy = element_vector("m", "sic", "ElectronCurrent")
jm = np.hypot(jx, jy)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax[0].plot(VD, ID / W, "o-", ms=3)
ax[0].plot(VD, VD * 1e3 / ron, "--", lw=0.8, label=f"V$_{{DS}}$ / R$_{{on,sp}}$, R$_{{on,sp}}$ = {ron:.3f} m$\\Omega$cm$^2$")
ax[0].set(xlabel="V$_{DS}$ (V)", ylabel="J$_D$ (A/cm$^2$)", title="(a) output, V$_{GS}$ = 18 V, 300 K",
          ylim=(0, 1.1 * ID.max() / W))
ax[0].legend()
c = ax[1].tripcolor(tri, facecolors=np.log10(np.maximum(jm, 1e-3)), cmap="inferno", vmin=0, vmax=5)
fig.colorbar(c, ax=ax[1], label="log$_{10}$ |J$_n$| (A/cm$^2$)")
X, Y = np.meshgrid(np.linspace(1.6, 3.45, 12), np.linspace(0.05, 2.4, 13))
k = tri.get_trifinder()(X, Y)
ok = (k >= 0) & (jm[k] > 10.0)              # no arrows in the depleted P-well (|J| < 10 A/cm^2)
U, V = jx[k[ok]], jy[k[ok]]
s = np.hypot(U, V) + 1e-30
ax[1].quiver(X[ok], Y[ok], -U / s, -V / s, color="c", angles="xy", scale_units="xy", scale=7, width=0.004)  # electrons move along -J
ax[1].set(xlim=(1.4, 3.5), ylim=(2.5, 0), xlabel="x (um)", ylabel="y (um)",
          title="(b) |J$_n$| and electron flow, V$_{GS}$ = 18 V, V$_{DS}$ = 5 V")
ax[1].grid(False)
savefig(fig, IMG / "output.png")
