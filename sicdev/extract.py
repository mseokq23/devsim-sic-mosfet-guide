"""Feature extraction from the transfer and output curves (fixed rules, decided before the runs).

Units: the 2D half-cell gives current per cm of device depth (A/cm).
Ron,sp uses the half-cell width:  Ron,sp = (dV/dI) * W_half  [ohm cm^2].
"""
from __future__ import annotations

import math
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def vg_at_current(vg, idd, i0):
    """First gate voltage where ID reaches i0 (interpolated in log ID). NaN if never reached."""
    vg, idd = _arr(vg), _arr(idd)
    above = np.where(idd >= i0)[0]
    if len(above) == 0 or above[0] == 0:
        return float("nan")
    k = above[0]
    i_lo, i_hi = idd[k - 1], idd[k]
    if i_lo <= 0:
        return float(vg[k])
    f = (math.log10(i0) - math.log10(i_lo)) / (math.log10(i_hi) - math.log10(i_lo))
    return float(vg[k - 1] + f * (vg[k] - vg[k - 1]))


def subthreshold_swing(vg, idd, lo, hi):
    """SS [mV/dec] = gate-voltage span between two currents / number of decades."""
    v_lo, v_hi = vg_at_current(vg, idd, lo), vg_at_current(vg, idd, hi)
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
        return float("nan")
    return 1e3 * (v_hi - v_lo) / math.log10(hi / lo)


def gm_max(vg, idd):
    vg, idd = _arr(vg), _arr(idd)
    if len(vg) < 3:
        return float("nan"), float("nan")
    gm = np.gradient(idd, vg)
    k = int(np.argmax(gm))
    return float(gm[k]), float(vg[k])


def current_at(x, y, x0):
    """Current at x0, interpolated in log (or linearly if any value <= 0)."""
    x, y = _arr(x), _arr(y)
    if x0 < x.min() - 1e-9 or x0 > x.max() + 1e-9:
        return float("nan")
    if np.all(y > 0):
        return float(10 ** np.interp(x0, x, np.log10(y)))
    return float(np.interp(x0, x, y))


def ron_sp(vds, idd, vmax, width_cm):
    """Low-VDS on-resistance from a least-squares line through the origin (VDS <= vmax)."""
    vds, idd = _arr(vds), _arr(idd)
    m = vds <= vmax + 1e-12
    if m.sum() < 2:
        return float("nan")
    g = float(np.sum(vds[m] * idd[m]) / np.sum(vds[m] ** 2))     # A/(cm V)
    return 1e3 * width_cm / g if g > 0 else float("nan")          # mOhm cm^2


def kcl_floor(idd, iss):
    """Numerical current floor: max |ID + IS| (current balance error of the discretisation)."""
    return float(np.max(np.abs(_arr(idd) + _arr(iss)))) if len(idd) else float("nan")


def extract_features(rec: dict, cfg: dict) -> dict:
    ex = cfg["extract"]
    tr, out = rec["curves"]["transfer"], rec["curves"]["output"]
    vg, idd = tr["vgs"], tr["id"]
    lo, hi = ex["ss_window_A_per_cm"]
    floor = kcl_floor(tr["id"], tr["is_"])
    gm, vgm = gm_max(vg, idd)
    va = rec.get("vg_at_current") or {}
    key = lambda t: f"{t:.3e}"
    vth_i, ss_i = vg_at_current(vg, idd, ex["icc_A_per_cm"]), subthreshold_swing(vg, idd, lo, hi)
    vth = va.get(key(ex["icc_A_per_cm"])) or vth_i            # bias-refined value if available
    ss = (1e3 * (va[key(hi)] - va[key(lo)]) / math.log10(hi / lo)
          if va.get(key(hi)) and va.get(key(lo)) else ss_i)
    f = dict(
        vth_V=vth, ss_mV_dec=ss, vth_interp_V=vth_i, ss_interp_mV_dec=ss_i,
        gm_max_S_per_cm=gm, vg_at_gm_max_V=vgm,
        ion_A_per_cm=current_at(vg, idd, ex["vgs_on"]),
        ron_mohm_cm2=ron_sp(out["vds"], out["id"], ex["ron_vds_max"], rec.get("width_cm", float("nan"))),
        id_vds_req_A_per_cm=current_at(out["vds"], out["id"], cfg["bias"].get("vds_required", 2.0))
        if out["id"] else float("nan"),
        kcl_floor_A_per_cm=floor,
    )
    for v in ex["feature_vgs"]:
        i = current_at(vg, idd, float(v))
        f[f"logid_vg{v:g}"] = math.log10(i) if i > 0 else float("nan")
    return f
