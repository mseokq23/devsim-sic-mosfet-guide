"""DC solve with backup/restore, and bias ramping with step halving.

Rule: if a bias step fails, go back to the last converged point and halve the step;
below min_step the run stops with ConvergenceError.
"""
from __future__ import annotations

import contextlib
import io
import time
import devsim as ds

from .physics import bias_name

SOLUTION_VARS = ("Potential", "Electrons", "Holes")
DEFAULT_TOL = dict(absolute_error=1e10, relative_error=1e-6, maximum_iterations=40)


class ConvergenceError(RuntimeError):
    pass


class SolveStats:
    def __init__(self):
        self.n_solves = 0
        self.n_fail = 0
        self.t_solve = 0.0

    def as_dict(self):
        return dict(n_solves=self.n_solves, n_fail=self.n_fail, t_solve_s=round(self.t_solve, 3))


STATS = SolveStats()
QUIET = True          # DEVSIM prints every Newton iteration to stdout; keep it in LOG instead
LOG = io.StringIO()   # log of the last solve (look here when something fails)


def quiet():
    global LOG
    if not QUIET:
        return contextlib.nullcontext()
    LOG = io.StringIO()
    return contextlib.redirect_stdout(LOG)


def _backup(dev):
    snap = {}
    for reg in ds.get_region_list(device=dev):
        names = ds.get_node_model_list(device=dev, region=reg)
        for v in SOLUTION_VARS:
            if v in names:
                snap[(reg, v)] = ds.get_node_model_values(device=dev, region=reg, name=v)
    return snap


def _restore(dev, snap):
    for (reg, v), vals in snap.items():
        ds.set_node_values(device=dev, region=reg, name=v, values=vals)


def solve_dc(dev, tol=None, restore_on_fail=True):
    tol = {**DEFAULT_TOL, **(tol or {})}
    snap = _backup(dev) if restore_on_fail else None
    t0 = time.time()
    STATS.n_solves += 1
    try:
        with quiet():
            info = ds.solve(type="dc", info=True, **tol)
        if isinstance(info, dict) and not info.get("converged", True):
            raise ds.error("Convergence failure (info)")
    except ds.error as e:
        STATS.n_fail += 1
        if snap is not None:
            _restore(dev, snap)
        raise ConvergenceError(str(e)) from e
    finally:
        STATS.t_solve += time.time() - t0


def ramp(dev, contact, target, step, min_step=1e-3, max_step=None, tol=None, callback=None, grow=1.5):
    """Move '<contact>_bias' to target. callback(v) is called after every converged point."""
    bn = bias_name(contact)
    v = ds.get_parameter(device=dev, name=bn)
    step = abs(step)
    max_step = max_step or step
    while abs(target - v) > 1e-12:
        s = 1.0 if target > v else -1.0
        nv = v + s * min(step, abs(target - v))
        ds.set_parameter(device=dev, name=bn, value=nv)
        try:
            solve_dc(dev, tol)
        except ConvergenceError:
            ds.set_parameter(device=dev, name=bn, value=v)
            step *= 0.5
            if step < min_step:
                raise ConvergenceError(f"{contact}: min step reached at {v:.4g} V -> {target:.4g} V")
            continue
        v = nv
        if callback:
            callback(v)
        step = min(step * grow, max_step)
    return v
