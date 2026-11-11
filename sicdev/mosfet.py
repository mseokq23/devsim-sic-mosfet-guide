"""One MOSFET run = one (device, temperature) with the fixed bias protocol:

  equilibrium -> VDS = vds_lin -> transfer VGS 0 -> 20 V -> VGS = vgs_on -> output VDS sweep

Every bias point starts from the previous solution; a failed step is halved (solver.ramp).
DEVSIM keeps global state, so for many runs start each one in a new process (sicdev.worker).
"""
from __future__ import annotations

import math
import time

import devsim as ds

from . import devices as dv, physics as ph, solver
from .config import material_from, gate_from
from .extract import extract_features
from .solver import ramp, ConvergenceError, SolveStats
from .utils import reset_devsim


def build(cfg, T, wjfet_scale=1.0, npwell_scale=1.0, qit_eff_cm2=None, mu_channel_scale=1.0,
          mesh_scale=None, extended=None, circuit=None, dev="m"):
    """Build the half-cell, set up the physics and solve the equilibrium (all contacts at 0 V)."""
    num = cfg["numerics"]
    ext = bool(num.get("extended_precision", False)) if extended is None else bool(extended)
    ms = float(cfg["mosfet"].get("mesh_scale", 1.0)) if mesh_scale is None else float(mesh_scale)
    ii = num.get("incomplete_ionization", True)
    holes = bool(num.get("holes", False))
    intf = dict(cfg["interface"])
    if qit_eff_cm2 is not None:
        intf["qit_eff_cm2"] = float(qit_eff_cm2)
    mat = material_from(cfg)
    mat.mu_surf300 *= float(mu_channel_scale)
    gate = gate_from(cfg)
    reset_devsim()
    ph.set_extended_precision(ext)
    with solver.quiet():
        lay = dv.build_mosfet(dev, cfg, wjfet_scale, npwell_scale, ms)
        dv.setup_physics(dev, lay, mat, T, ii=ii, interface=intf, gate=gate, circuit=circuit)
    with solver.quiet():          # DEVSIM reports every replaced equation here; not needed in the logs
        dv.equilibrate(dev, lay, ii=ii, tol=num["tol"], holes=holes, circuit=circuit)
    lay["info"].update(mesh_scale=ms, extended=ext, holes=holes)
    return lay


def run_mosfet(cfg, T, wjfet_scale=1.0, npwell_scale=1.0, qit_eff_cm2=None, mu_channel_scale=1.0,
               mesh_scale=None, extended=None, output_curve=True) -> dict:
    b, tol = cfg["bias"], cfg["numerics"]["tol"]
    solver.STATS = SolveStats()
    t0 = time.time()
    qit = float(cfg["interface"]["qit_eff_cm2"] if qit_eff_cm2 is None else qit_eff_cm2)
    rec = dict(temperature_K=float(T), wjfet_scale=float(wjfet_scale), npwell_scale=float(npwell_scale),
               qit_eff_cm2=qit, mu_channel_scale=float(mu_channel_scale),
               converged=False, error=None, stage="build")
    tr = dict(vgs=[], id=[], is_=[])
    out = dict(vgs=float(b["vgs_on"]), vds=[], id=[])
    try:
        lay = build(cfg, T, wjfet_scale, npwell_scale, qit, mu_channel_scale, mesh_scale, extended)
        rec.update(n_nodes=lay["info"]["n_nodes"], width_cm=lay["info"]["width_cm"],
                   mesh_scale=lay["info"]["mesh_scale"], extended=lay["info"]["extended"])
        rec["stage"] = "vds_lin"
        ramp("m", "drain", b["vds_lin"], 0.05, tol=tol)

        ex = cfg["extract"]
        targets = sorted({*ex["ss_window_A_per_cm"], ex["icc_A_per_cm"]})
        vg_at = {}

        def refine(t, v0, i0, v1, i1, iters=4):
            """Secant search in log10(ID) between two transfer points -> VGS where ID = t.
            Removes the interpolation error from Vth and SS; these solves are not added to the curve."""
            lt = math.log10(t)
            for _ in range(iters):
                if i0 > 0 and i1 > 0:
                    f = (lt - math.log10(i0)) / (math.log10(i1) - math.log10(i0))
                    v = v0 + min(max(f, 0.02), 0.98) * (v1 - v0)
                else:
                    v = 0.5 * (v0 + v1)
                ds.set_parameter(device="m", name=ph.bias_name("gate"), value=v)
                try:
                    solver.solve_dc("m", tol)
                except ConvergenceError:
                    return None
                i = ph.contact_current("m", "drain")
                if i > 0 and abs(math.log10(i) - lt) < 1e-5:
                    return v
                if i < t:
                    v0, i0 = v, i
                else:
                    v1, i1 = v, i
            return v

        def rec_tr(v):
            tr["vgs"].append(float(v))
            tr["id"].append(ph.contact_current("m", "drain"))
            tr["is_"].append(ph.contact_current("m", "source"))
            if len(tr["vgs"]) > 1:
                v0, i0, i1 = tr["vgs"][-2], tr["id"][-2], tr["id"][-1]
                for t in targets:
                    if f"{t:.3e}" not in vg_at and i0 < t <= i1:
                        vg_at[f"{t:.3e}"] = refine(t, v0, i0, float(v), i1)

        rec_tr(0.0)
        rec["stage"] = "transfer"
        for stop, step in b["vgs_segments"]:
            ramp("m", "gate", stop, step, tol=tol, callback=rec_tr)
        if output_curve:
            rec["stage"] = "output"
            ramp("m", "gate", b["vgs_on"], 0.5, tol=tol)

            def rec_out(v):
                out["vds"].append(float(v))
                out["id"].append(ph.contact_current("m", "drain"))

            rec_out(b["vds_lin"])
            try:
                for stop, step in b["vds_segments"]:
                    ramp("m", "drain", stop, step, tol=tol, callback=rec_out)
            except ConvergenceError as e:
                # the output curve may stop early once it has passed vds_required
                if not out["vds"] or max(out["vds"]) < b.get("vds_required", 2.0) - 1e-9:
                    raise
                rec["output_truncated_at_V"] = max(out["vds"])
        rec["converged"] = True
        rec["stage"] = "done"
        rec["vg_at_current"] = vg_at
    except ConvergenceError as e:
        rec["error"] = f"CONVERGENCE@{rec['stage']}: {e}"[:300]
    rec["runtime_s"] = round(time.time() - t0, 3)
    rec.update(solver.STATS.as_dict())
    rec["curves"] = dict(transfer=tr, output=out)
    rec["features"] = extract_features(rec, cfg) if rec["converged"] else {}
    return rec
