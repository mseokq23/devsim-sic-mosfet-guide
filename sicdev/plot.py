"""Small matplotlib helpers shared by the step scripts (figures go to docs/img/NN/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import devsim as ds

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110,
                     "savefig.dpi": 140, "legend.fontsize": 8})


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("saved", path.relative_to(Path(__file__).resolve().parents[1]))


def triangulation(dev, reg, um=True):
    """matplotlib Triangulation of a 2D DEVSIM region (coordinates in um by default)."""
    s = 1e4 if um else 1.0
    x = np.array(ds.get_node_model_values(device=dev, region=reg, name="x")) * s
    y = np.array(ds.get_node_model_values(device=dev, region=reg, name="y")) * s
    tri = np.array(ds.get_element_node_list(device=dev, region=reg))
    return mtri.Triangulation(x, y, tri)


def element_vector(dev, reg, edge_model):
    """Per-triangle vector (x, y) of an edge model such as ElectronCurrent (mean of the 3 element edges)."""
    ds.element_from_edge_model(edge_model=edge_model, device=dev, region=reg)
    vx = np.array(ds.get_element_model_values(device=dev, region=reg, name=f"{edge_model}_x"))
    vy = np.array(ds.get_element_model_values(device=dev, region=reg, name=f"{edge_model}_y"))
    return vx.reshape(-1, 3).mean(axis=1), vy.reshape(-1, 3).mean(axis=1)


def signed_log(v, floor=1e14):
    """sign(v) * log10(|v|/floor) clipped at 0, for net doping maps."""
    v = np.asarray(v, float)
    return np.sign(v) * np.log10(np.maximum(np.abs(v), floor) / floor)
