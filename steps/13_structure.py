"""13. Cross-section of the simulated half-cell (planar 4H-SiC MOSFET), drawn from configs/baseline.yaml.

  python steps/13_structure.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matplotlib.patches import Rectangle, FancyArrowPatch
from sicdev.config import load_config
from sicdev.devices import mosfet_geometry
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "13"
cfg = load_config()
g = mosfet_geometry(cfg)
dop = cfg["mosfet"]["doping"]
xh, tox = g["x_half_um"], g["tox_um"]
L_ch = g["x_pw_um"] - g["x_ns1_um"]
print(f"half-cell width {xh:.2f} um (cell pitch {2 * xh:.1f} um), total depth {g['y_max_um']:.1f} um")
print(f"channel length {L_ch:.2f} um, JFET width {g['wjfet_um']:.2f} um (half-cell {g['wjfet_um'] / 2:.2f} um)")
print(f"oxide {tox * 1e3:.0f} nm from x = {g['x_ox_start_um']} um, source contact x = 0 - {g['x_src_contact_um']} um")

COL = dict(n_drift="#cfe3f5", n_jfet="#a9cdee", n_plus="#4f8fd1", n_sub="#2f6fb5", p_well="#f5c8c3",
           p_plus="#d9534f", oxide="#c8c8c8", metal="#555555")


def draw(ax, ylim):
    boxes = [  # (x0, x1, y0, y1, color)
        (0, xh, 0, g["y_sub_um"], COL["n_drift"]),
        (0, xh, g["y_sub_um"], g["y_max_um"], COL["n_sub"]),
        (g["x_pw_um"], xh, 0, g["d_jfet_um"], COL["n_jfet"]),
        (0, g["x_pw_um"], 0, g["d_pw_um"], COL["p_well"]),
        (0, g["x_pp_um"], 0, g["d_pp_um"], COL["p_plus"]),
        (g["x_ns0_um"], g["x_ns1_um"], 0, g["d_ns_um"], COL["n_plus"]),
        (g["x_ox_start_um"], xh, -tox, 0, COL["oxide"]),
    ]
    for x0, x1, y0, y1, c in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="k", lw=0.5))
    t = 0.012 * (ylim[0] - ylim[1])
    ax.add_patch(Rectangle((g["x_ox_start_um"], -tox - t), xh - g["x_ox_start_um"], t, color=COL["metal"]))
    ax.add_patch(Rectangle((0, -t), g["x_src_contact_um"], t, color=COL["metal"]))
    ax.add_patch(Rectangle((0, g["y_max_um"]), xh, t, color=COL["metal"]))
    ax.set(xlim=(-0.05, xh + 0.05), ylim=ylim, xlabel="x (um)", ylabel="y (um)")
    ax.grid(False)


def label(ax, x, y, txt, **kw):
    ax.text(x, y, txt, ha="center", va="center", fontsize=kw.pop("fontsize", 7), **kw)


fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
draw(ax[0], (g["y_max_um"] + 0.6, -0.6))
label(ax[0], xh / 2, 5.0, f"n- drift {dop['N_drift']:.0e} cm$^{{-3}}$, 10 um", fontsize=8)
label(ax[0], xh / 2, 10.5, f"n+ substrate {dop['N_sub']:.0e}", color="w", fontsize=8)
label(ax[0], xh / 2, g["y_max_um"] + 0.4, "drain", fontsize=8)
label(ax[0], xh / 2, -0.38, "source / gate (zoom in b)", fontsize=8)
ax[0].set_title("(a) whole half-cell")

draw(ax[1], (1.35, -0.25))
label(ax[1], 0.25, 0.15, f"p+\n{dop['N_pp']:.0e}", color="w")
label(ax[1], 1.15, 0.125, f"n+ source {dop['N_ns']:.0e}", color="w")
label(ax[1], 1.15, 0.47, f"P-well {dop['N_pw']:.0e}")
label(ax[1], 2.9, 0.62, f"JFET region\n{dop['N_jfet']:.0e}")
label(ax[1], 1.15, 1.02, f"n- drift {dop['N_drift']:.0e}")
label(ax[1], 0.7, -0.075, "source contact", fontsize=8)
label(ax[1], 2.5, -0.12, "gate (n+ poly)", fontsize=8)
label(ax[1], 2.5, -0.025, "SiO$_2$ 50 nm")
ax[1].annotate("", xy=(g["x_pw_um"], 0.035), xytext=(g["x_ns1_um"], 0.035),
               arrowprops=dict(arrowstyle="<->", lw=0.8))
label(ax[1], 0.5 * (g["x_ns1_um"] + g["x_pw_um"]), 0.09, f"channel {L_ch:.1f} um")
for p0, p1 in (((1.4, 0.12), (1.85, 0.02)), ((2.3, 0.02), (2.95, 0.25)), ((2.95, 0.25), (2.95, 1.3))):
    ax[1].add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color="tab:green", lw=1.6))
ax[1].text(3.05, 0.95, "electron\npath", color="tab:green", fontsize=8)
for xs in (0.0, xh):
    ax[1].axvline(xs, color="k", ls="--", lw=0.8)
ax[1].text(0.03, 1.31, "symmetry line (source side)", fontsize=7, va="bottom")
ax[1].text(xh - 0.03, 1.31, "symmetry line (JFET center)", fontsize=7, va="bottom", ha="right")
ax[1].set_title("(b) top 1.3 um")
savefig(fig, IMG / "halfcell.png")
