"""14. 2D mesh of the half-cell: mesh lines, regions, contacts, interface; node counts per mesh_scale.

  python steps/14_mesh.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import devsim as ds
from sicdev.config import load_config
from sicdev.devices import build_mosfet, mosfet_geometry, mosfet_mesh_spec
from sicdev import solver
from sicdev.utils import reset_devsim
from sicdev.plot import plt, savefig, triangulation

IMG = ROOT / "docs" / "img" / "14"
cfg = load_config()
g = mosfet_geometry(cfg)
xspec, yspec, dg = mosfet_mesh_spec(g)
print("x mesh lines (position um: spacing um):", ", ".join(f"{p:g}: {s:g}" for p, s in xspec))
print("y mesh lines (position um: spacing um):", ", ".join(f"{p:g}: {s:g}" for p, s in yspec))

for ms in (2.0, 1.5, 1.0):
    reset_devsim()
    with solver.quiet():
        lay = build_mosfet("m", cfg, mesh_scale=ms)
    print(f"mesh_scale {ms}: {lay['info']['n_nodes']} nodes (SiC + oxide), "
          f"{len(ds.get_element_node_list(device='m', region='sic'))} triangles in SiC")

reset_devsim()
with solver.quiet():
    lay = build_mosfet("m", cfg, mesh_scale=1.5)
print("regions :", ds.get_region_list(device="m"))
print("contacts:", ds.get_contact_list(device="m"))
print("interface:", ds.get_interface_list(device="m"))
ts, to = triangulation("m", "sic"), triangulation("m", "oxide")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1.9]))
for a in ax:
    a.triplot(ts, lw=0.25, color="tab:blue")
    a.triplot(to, lw=0.25, color="tab:orange")
    a.invert_yaxis()
    a.grid(False)
ax[0].set(xlim=(0, g["x_half_um"]), ylim=(g["y_max_um"], -0.3), xlabel="x (um)", ylabel="y (um)",
          title="(a) whole half-cell")
ax[1].set(xlim=(1.3, 2.9), ylim=(0.3, -0.06), xlabel="x (um)", ylabel="y (um)",
          title="(b) channel: n+ edge (x=1.8) to P-well edge (x=2.3)")
for xl in (g["x_ns1_um"], g["x_pw_um"]):
    ax[1].axvline(xl, color="k", lw=0.8, ls="--")
ax[1].axhline(0, color="k", lw=0.8)
savefig(fig, IMG / "mesh.png")
