"""16. Half-cell in equilibrium (all contacts 0 V): Newton log, potential and electron density maps,
and why the 2D runs solve electrons only (holes=False).

  python steps/16_equilibrium.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import devsim as ds
from sicdev.config import load_config, deep
from sicdev import mosfet, physics as ph, solver
from sicdev.plot import plt, savefig, triangulation

IMG = ROOT / "docs" / "img" / "16"
cfg = load_config()
tol = cfg["numerics"]["tol"]

print("--- equilibrium: Poisson only, then electron drift-diffusion (Newton log) ---")
solver.QUIET = False
t0 = time.time()
lay = mosfet.build(cfg, 300.0)
solver.QUIET = True
print(f"--- end ({time.time() - t0:.2f} s) ---\n")

g = lay["info"]["geometry"]
x = ph.node_array("m", "sic", "x") * 1e4
y = ph.node_array("m", "sic", "y") * 1e4
psi = ph.node_array("m", "sic", "Potential")
n = ph.node_array("m", "sic", "Electrons")
p = ph.node_array("m", "sic", "IntrinsicHoles")
print(f"nodes {lay['info']['n_nodes']}, Potential {psi.min():+.3f} .. {psi.max():+.3f} V")
print(f"electrons {n.min():.2e} .. {n.max():.2e} cm^-3")
print(f"holes (p = ni exp(-Potential/Vt)) {p.min():.2e} .. {p.max():.2e} cm^-3")

tri = triangulation("m", "sic")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
c0 = ax[0].tripcolor(tri, psi, cmap="viridis", shading="gouraud")
fig.colorbar(c0, ax=ax[0], label="Potential (V)")
c1 = ax[1].tripcolor(tri, np.log10(np.maximum(n, 1e-40)), cmap="magma", shading="gouraud", vmin=-40, vmax=20)
fig.colorbar(c1, ax=ax[1], label="log$_{10}$ n (cm$^{-3}$)")
for a, t in zip(ax, ("Potential, equilibrium", "electron density, equilibrium")):
    a.set(xlim=(0, g["x_half_um"]), ylim=(1.6, 0), xlabel="x (um)", ylabel="y (um)", title=t)
    a.grid(False)
savefig(fig, IMG / "equilibrium_maps.png")

# electron-only (holes=False) vs electrons + holes, same bias points
print("\nelectron-only vs electrons + holes (VDS = 0.1 V):")
print(f"{'model':<18} {'ID at VGS=4 V (A/cm)':>21} {'ID at VGS=10 V (A/cm)':>22} {'solves':>7} {'time (s)':>9}")
res = {}
for holes in (False, True):
    c = deep(cfg)
    c["numerics"]["holes"] = holes
    solver.STATS = solver.SolveStats()
    t0 = time.time()
    mosfet.build(c, 300.0)
    solver.ramp("m", "drain", 0.1, 0.05, tol=tol)
    solver.ramp("m", "gate", 4.0, 0.25, tol=tol)
    i4 = ph.contact_current("m", "drain")
    solver.ramp("m", "gate", 10.0, 0.25, tol=tol)
    i10 = ph.contact_current("m", "drain")
    res[holes] = (i4, i10)
    name = "electrons + holes" if holes else "electrons only"
    print(f"{name:<18} {i4:21.10e} {i10:22.10e} {solver.STATS.n_solves:7d} {time.time() - t0:9.1f}")
    if holes:
        ih = ds.get_contact_current(device="m", contact="drain", equation="HoleContinuityEquation")
        print(f"{'':18} hole current at the drain (VGS = 10 V): {ih:.1e} A/cm")
same = res[True] == res[False]
print("drain currents of the two models are " + ("bit-for-bit identical" if same else
      f"different: {abs(res[True][0] / res[False][0] - 1):.1e} (4 V), {abs(res[True][1] / res[False][1] - 1):.1e} (10 V)"))
