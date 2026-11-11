"""15. Doping, interface sheet charge and surface-mobility factor on the 2D mesh.

  python steps/15_doping.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config
from sicdev.devices import build_mosfet
from sicdev import physics as ph, solver
from sicdev.params import Q
from sicdev.plot import plt, savefig, triangulation, signed_log

IMG = ROOT / "docs" / "img" / "15"
cfg = load_config()
with solver.quiet():
    lay = build_mosfet("m", cfg, mesh_scale=1.5)
g = lay["info"]["geometry"]
don, acc = lay["semis"]["sic"]
x = ph.node_array("m", "sic", "x") * 1e4
y = ph.node_array("m", "sic", "y") * 1e4
vol = ph.node_array("m", "sic", "NodeVolume")
net = don - acc
il, sf = lay["intf_len"], lay["surf"]

print(f"interface nodes: {np.count_nonzero(il)}, sum of IntfLen = {il.sum() * 1e4:.4f} um "
      f"(oxide length {g['x_half_um'] - g['x_ox_start_um']:.4f} um)")
sig = Q * (cfg["interface"]["qf_cm2"] + cfg["interface"]["qit_eff_cm2"])
print(f"sheet charge q(Qf + Qit) = {sig:.3e} C/cm^2 (Qf = {cfg['interface']['qf_cm2']:.1e}, "
      f"Qit = {cfg['interface']['qit_eff_cm2']:.1e} cm^-2)")
k = np.argmax(il)
print(f"example interface node x = {x[k]:.4f} um: IntfLen = {il[k] * 1e4 * 1e3:.2f} nm, "
      f"NodeVolume = {vol[k]:.3e} cm^2, sheet -> volume factor IntfLen/NodeVolume = {il[k] / vol[k]:.3e} 1/cm")
for xc, name in ((1.0, "n+ source / P-well"), (2.05, "channel center"), (3.0, "JFET region")):
    sel = np.argsort(np.abs(x - xc) + 0 * y)
    col = sel[np.abs(x[sel] - x[sel[0]]) < 1e-9]
    col = col[np.argsort(y[col])]
    vals = ", ".join(f"y={yy:.2f}: {nn:+.1e}" for yy, nn in zip(y[col], net[col]) if yy in (0.0,) or
                     any(abs(yy - t) < 0.004 for t in (0.1, 0.3, 0.5, 0.8, 1.5)))
    print(f"net doping at x = {x[col[0]]:.3f} um ({name}): {vals}")

tri = triangulation("m", "sic")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
cs = ax[0].tripcolor(tri, signed_log(net), cmap="RdBu_r", shading="gouraud", vmin=-6, vmax=6)
ax[0].set(xlim=(0, g["x_half_um"]), ylim=(1.6, -0.02), xlabel="x (um)", ylabel="y (um)",
          title="net doping N$_D$ - N$_A$ (signed log)")
cb = fig.colorbar(cs, ax=ax[0], ticks=range(-6, 7, 2))
cb.ax.set_yticklabels(["-1e20", "-1e18", "-1e16", "0", "1e16", "1e18", "1e20"])
cb.set_label("cm$^{-3}$ (red n-type, blue p-type)")
ax[0].grid(False)
for xc, ls in ((1.0, "-"), (2.05, "--"), (3.0, ":")):
    sel = np.abs(x - x[np.argmin(np.abs(x - xc))]) < 1e-9
    o = np.argsort(y[sel])
    ax[1].semilogy(y[sel][o], np.abs(net[sel][o]), ls, label=f"x = {x[np.argmin(np.abs(x - xc))]:.2f} um")
    ax[0].axvline(x[np.argmin(np.abs(x - xc))], color="k", ls=ls, lw=0.8)
ax[1].set(xlim=(0, 1.6), ylim=(1e15, 3e19), xlabel="y (um)", ylabel="|N$_D$ - N$_A$| (cm$^{-3}$)",
          title="vertical cuts")
ax[1].legend()
savefig(fig, IMG / "doping.png")

fig, ax = plt.subplots(figsize=(5.4, 3.4))
xc = x[np.argmin(np.abs(x - 2.05))]
m = np.abs(x - xc) < 1e-9
o = np.argsort(y[m])
ax.plot(y[m][o] * 1e3, sf[m][o], ".-", label="SurfFactor at the mesh nodes")
ax.plot(np.linspace(0, 30, 301), np.exp(-np.linspace(0, 30, 301) / (cfg["material"]["surf_lambda_um"] * 1e3)), "k:", lw=0.8, label="exp(-y / 3 nm)")
ax.set(xlabel="depth (nm)", ylabel="SurfFactor", xlim=(0, 30), title=f"surface mobility weight (x = {xc:.2f} um)")
ax.legend()
savefig(fig, IMG / "surf_factor.png")
