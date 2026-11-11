from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import devsim as ds


def reset_devsim():
    """Delete every device and mesh so the next simulation starts clean."""
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
        try:
            ds.delete_mesh(mesh=d + "_mesh")
        except ds.error:
            pass


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, cls=NpEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def env_info() -> dict:
    info = ds.get_parameter(name="info")
    return dict(devsim=info.get("version"), solver=info.get("direct_solver"),
                extended=info.get("extended_precision"), python=platform.python_version())
