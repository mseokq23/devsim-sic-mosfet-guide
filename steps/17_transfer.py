"""17. Transfer curve ID-VGS at VDS = 0.1 V, 300 K, with the bias ramp of the fixed protocol,
plus the electron density under the gate while the channel turns on.

  python steps/17_transfer.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config
from sicdev import mosfet, physics as ph, solver
from sicdev.plot import plt, savefig, triangulation

IMG = ROOT / "docs" / "img" / "17"
cfg = load_config()
b, tol = cfg["bias"], cfg["numerics"]["tol"]
t0 = time.time()
lay = mosfet.build(cfg, 300.0)
W = lay["info"]["width_cm"]
solver.ramp("m", "drain", b["vds_lin"], 0.05, tol=tol)
x = ph.node_array("m", "sic", "x") * 1e4
y = ph.node_array("m", "sic", "y") * 1e4
col = np.abs(x - x[np.argmin(np.abs(x - 2.05))]) < 1e-9          # vertical line at the channel center
order = np.argsort(y[col])
VG, ID, IS, prof, maps = [], [], [], {}, {}


def record(v):
    VG.append(v)
    ID.append(ph.contact_current("m", "drain"))
    IS.append(ph.contact_current("m", "source"))
    if any(abs(v - t) < 1e-9 for t in (0, 2, 4, 6, 10, 18)):
        prof[v] = ph.node_array("m", "sic", "Electrons")[col][order]
    if any(abs(v - t) < 1e-9 for t in (0, 4, 18)):
        maps[v] = ph.node_array("m", "sic", "Electrons")


record(0.0)
for stop, step in b["vgs_segments"]:
    solver.ramp("m", "gate", stop, step, tol=tol, callback=record)
VG, ID, IS = map(np.array, (VG, ID, IS))
floor = np.max(np.abs(ID + IS))
print(f"{len(VG)} points, {solver.STATS.n_solves} solves ({solver.STATS.n_fail} failed), {time.time() - t0:.1f} s")
print(f"current balance |ID + IS| max = {floor:.1e} A/cm  (numerical floor of the current)")
print(f"{'VGS (V)':>8} {'ID (A/cm)':>11} {'JD (A/cm2)':>11}")
for v in (0, 2, 3, 4, 5, 6, 8, 10, 14, 18, 20):
    k = int(np.argmin(np.abs(VG - v)))
    print(f"{VG[k]:8.2f} {ID[k]:11.3e} {ID[k] / W:11.3e}")
print(f"half-cell width W = {W * 1e4:.2f} um -> JD = ID / W")
depth = y[col][order] * 1e3
print("electrons at the channel center (x = 2.05 um):")
for v, n in sorted(prof.items()):
    print(f"  VGS = {v:4.1f} V: surface {n[0]:.2e}, 2 nm {np.interp(2, depth, n):.2e}, "
          f"5 nm {np.interp(5, depth, n):.2e}, 10 nm {np.interp(10, depth, n):.2e} cm^-3")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
ax[0].semilogy(VG, np.abs(ID), ".-", ms=3)
ax[0].axhline(floor, color="tab:red", lw=0.8, ls="--")
ax[0].text(12, floor * 3, "numerical floor |ID + IS|", color="tab:red", fontsize=8)
ax[0].set(xlabel="V$_{GS}$ (V)", ylabel="|I$_D$| (A/cm)", title="(a) transfer, V$_{DS}$ = 0.1 V, 300 K",
          ylim=(1e-30, 1e-1))
ax[1].plot(VG, ID * 1e3, ".-", ms=3)
ax[1].set(xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (mA/cm)", title="(b) linear scale")
sec = ax[1].secondary_yaxis("right", functions=(lambda i: i * 1e-3 / W, lambda j: j * W * 1e3))
sec.set_ylabel("J$_D$ (A/cm$^2$)")
for v, n in sorted(prof.items()):
    ax[2].semilogy(y[col][order] * 1e3, n, label=f"V$_{{GS}}$ = {v:g} V")
ax[2].set(xlim=(0, 60), ylim=(1e-30, 1e21), xlabel="depth at x = 2.05 um (nm)", ylabel="n (cm$^{-3}$)",
          title="(c) electrons at the channel center")
ax[2].legend(fontsize=7)
savefig(fig, IMG / "transfer.png")

tri = triangulation("m", "sic")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
for a, v in zip(ax, (0, 4, 18)):
    c = a.tripcolor(tri, np.log10(np.maximum(maps[v], 1e-10)), cmap="magma", shading="gouraud", vmin=0, vmax=20)
    a.set(xlim=(1.6, 2.6), ylim=(0.05, 0), xlabel="x (um)", ylabel="y (um)",
          title=f"log$_{{10}}$ n, V$_{{GS}}$ = {v} V (top 50 nm)")
    for xl in (1.8, 2.3):
        a.axvline(xl, color="w", lw=0.6, ls="--")
    a.grid(False)
fig.colorbar(c, ax=ax, label="log$_{10}$ n (cm$^{-3}$)", shrink=0.9)
fig.savefig(IMG / "electrons_on.png", dpi=140)
plt.close(fig)
print("saved docs/img/17/electrons_on.png")
