"""Device builders: 1D PN diode, 1D MOS capacitor, 2D half-cell planar SiC MOSFET.

Geometry inputs are in micrometers; DEVSIM works in cm (1 um = 1e-4 cm).
Doping profiles are boxes smoothed with erf (straggle ~ lateral spread of an implant).

Every builder returns a "layout" dict that setup_physics() uses:
  semis      {region: (donors, acceptors)}   node arrays (cm^-3)
  ohmic      [(region, contact)]
  oxides     [region]
  gates      [(oxide region, contact)]
  interfaces [interface name]
  intf_len   interface length per node (cm) for the sheet charge, or None
  surf       surface factor exp(-depth/lambda) for the channel mobility, or None
"""
from __future__ import annotations

import numpy as np
import devsim as ds
from scipy.special import erf

from . import physics as ph
from .params import SiC4H, SiO2, Gate
from .solver import solve_dc

UM = 1e-4
BIG = 1e3   # "infinity" in um for half-open boxes


def erfbox(x, y, x0, x1, y0, y1, s):
    """Smoothed indicator of [x0,x1] x [y0,y1] (same unit for all)."""
    fx = 0.5 * (erf((x - x0) / s) - erf((x - x1) / s))
    fy = 0.5 * (erf((y - y0) / s) - erf((y - y1) / s)) if y is not None else 1.0
    return fx * fy


def _lines(mesh, direction, spec, scale):
    """Mesh lines: (position um, spacing um). scale multiplies every spacing (mesh study)."""
    for pos_um, ps_um in spec:
        ds.add_2d_mesh_line(mesh=mesh, dir=direction, pos=pos_um * UM, ps=ps_um * UM * scale)


# --------------------------------------------------------------- 1D PN diode
def build_pn_diode(dev, cfg):
    g = cfg["pn_diode"]
    L, xj = g["length_um"], g["xj_um"]
    mesh = dev + "_mesh"
    ds.create_1d_mesh(mesh=mesh)
    spec = [(0.0, 0.05), (xj - 0.5, 0.01), (xj, 0.002), (xj + 1.5, 0.02), (L, 0.2)]
    for pos, ps in spec:
        ds.add_1d_mesh_line(mesh=mesh, pos=pos * UM, ps=ps * UM, tag=f"t{pos}")
    ds.add_1d_region(mesh=mesh, material="SiC", region="sic", tag1="t0.0", tag2=f"t{L}")
    ds.add_1d_contact(mesh=mesh, name="anode", tag="t0.0", material="metal")
    ds.add_1d_contact(mesh=mesh, name="cathode", tag=f"t{L}", material="metal")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    x = ph.node_array(dev, "sic", "x") / UM
    s = g.get("straggle_um", 0.01)
    acc = g["NA"] * erfbox(x, None, -BIG, xj, 0, 0, s)
    don = g["ND"] * erfbox(x, None, xj, BIG, 0, 0, s)
    return dict(semis={"sic": (don, acc)}, ohmic=[("sic", "anode"), ("sic", "cathode")],
                oxides=[], gates=[], interfaces=[], intf_len=None, surf=None)


# ------------------------------------------------------------- 1D MOS capacitor
def build_moscap(dev, cfg):
    g = cfg["moscap"]
    tox, L = g["tox_um"], g["length_um"]
    mesh = dev + "_mesh"
    ds.create_1d_mesh(mesh=mesh)
    spec = [(-tox, tox / 10), (0.0, 0.0005), (0.05, 0.003), (0.5, 0.02), (L, 0.1)]
    for pos, ps in spec:
        ds.add_1d_mesh_line(mesh=mesh, pos=pos * UM, ps=ps * UM, tag=f"t{pos}")
    ds.add_1d_region(mesh=mesh, material="Oxide", region="oxide", tag1=f"t{-tox}", tag2="t0.0")
    ds.add_1d_region(mesh=mesh, material="SiC", region="sic", tag1="t0.0", tag2=f"t{L}")
    ds.add_1d_contact(mesh=mesh, name="gate", tag=f"t{-tox}", material="metal")
    ds.add_1d_contact(mesh=mesh, name="body", tag=f"t{L}", material="metal")
    ds.add_1d_interface(mesh=mesh, name="ox_sic", tag="t0.0")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    x = ph.node_array(dev, "sic", "x") / UM
    acc = np.full_like(x, g["NA"])
    don = np.zeros_like(x)
    intf = np.where(np.abs(x) < 1e-9, 1.0, 0.0)      # 1D: sheet charge per unit area on the surface node
    surf = np.exp(-np.maximum(x, 0) / cfg["material"].get("surf_lambda_um", 0.003))
    return dict(semis={"sic": (don, acc)}, ohmic=[("sic", "body")], oxides=["oxide"],
                gates=[("oxide", "gate")], interfaces=["ox_sic"], intf_len=intf, surf=surf)


# ------------------------------------------------- 2D half-cell planar MOSFET
def mosfet_geometry(cfg, wjfet_scale=1.0):
    """Half-cell geometry (um).  The cell pitch is fixed; a wider JFET region moves the
    self-aligned P-well / n+ source edge by -dW/2, so the channel length and the half-cell
    width stay the same."""
    g = dict(cfg["mosfet"]["geometry"])
    w0 = g["wjfet_um"]
    g["wjfet_um"] = w0 * wjfet_scale
    shift = 0.5 * (g["wjfet_um"] - w0)        # > 0: wider JFET, narrower P-well
    g["x_half_um"] = g["x_pw_um"] + 0.5 * w0  # fixed half-pitch
    g["x_pw_um"] -= shift
    g["x_ns1_um"] -= shift
    if not (g["x_ox_start_um"] < g["x_ns1_um"] < g["x_pw_um"] < g["x_half_um"]
            and g["x_src_contact_um"] < g["x_ns1_um"]):
        raise ValueError(f"invalid half-cell geometry for wjfet_scale={wjfet_scale}: {g}")
    g["y_sub_um"] = g["t_drift_um"]
    g["y_max_um"] = g["t_drift_um"] + g["t_sub_um"]
    return g


def mosfet_doping(x, y, g, dop, npwell_scale=1.0):
    """Donor and acceptor arrays (cm^-3) on node coordinates x, y (um)."""
    s = dop.get("straggle_um", 0.03)
    acc = (dop["N_pw"] * npwell_scale * erfbox(x, y, -BIG, g["x_pw_um"], -BIG, g["d_pw_um"], s)
           + dop["N_pp"] * erfbox(x, y, -BIG, g["x_pp_um"], -BIG, g["d_pp_um"], s))
    don = (dop["N_drift"]
           + (dop["N_jfet"] - dop["N_drift"]) * erfbox(x, y, g["x_pw_um"], BIG, -BIG, g["d_jfet_um"], s)
           + dop["N_ns"] * erfbox(x, y, g["x_ns0_um"], g["x_ns1_um"], -BIG, g["d_ns_um"], s)
           + dop["N_sub"] * erfbox(x, y, -BIG, BIG, g["y_sub_um"], BIG, s))
    return don, acc


def mosfet_mesh_spec(g):
    tox, xh, ym = g["tox_um"], g["x_half_um"], g["y_max_um"]
    xch0, xch1 = g["x_ns1_um"], g["x_pw_um"]
    xspec = [(0.0, 0.10), (g["x_pp_um"], 0.03), (g["x_src_contact_um"], 0.02), (g["x_ox_start_um"], 0.02),
             (xch0, 0.008), (0.5 * (xch0 + xch1), 0.025), (xch1, 0.008), (min(xch1 + 0.4, xh - 0.05), 0.05),
             (xh, 0.08)]
    yspec = [(-tox, tox / 6), (0.0, 0.0008), (0.01, 0.002), (0.05, 0.008),
             (g["d_ns_um"], 0.015), (g["d_pp_um"], 0.02), (g["d_pw_um"], 0.015),
             (g["d_jfet_um"], 0.04), (2.5, 0.25), (g["y_sub_um"], 0.05), (ym, 0.25)]
    # thin "gas" strips (regions without equations) above and below the device: DEVSIM's 2D
    # mesher puts a contact only on a boundary between two regions
    dg = 1e-3
    yspec += [(-tox - dg, dg), (ym + dg, dg)]
    xspec = sorted({round(p, 6): s for p, s in xspec}.items())
    yspec = sorted({round(p, 6): s for p, s in yspec}.items())
    return xspec, yspec, dg


def build_mosfet(dev, cfg, wjfet_scale=1.0, npwell_scale=1.0, mesh_scale=1.0):
    g = mosfet_geometry(cfg, wjfet_scale)
    dop = cfg["mosfet"]["doping"]
    tox, xh, ym = g["tox_um"], g["x_half_um"], g["y_max_um"]
    xo, xs = g["x_ox_start_um"], g["x_src_contact_um"]
    xspec, yspec, dg = mosfet_mesh_spec(g)
    mesh = dev + "_mesh"
    ds.create_2d_mesh(mesh=mesh)
    _lines(mesh, "x", xspec, mesh_scale)
    _lines(mesh, "y", yspec, mesh_scale)
    ds.add_2d_region(mesh=mesh, material="SiC", region="sic", xl=0, xh=xh * UM, yl=0, yh=ym * UM)
    ds.add_2d_region(mesh=mesh, material="Oxide", region="oxide", xl=xo * UM, xh=xh * UM, yl=-tox * UM, yh=0)
    ds.add_2d_region(mesh=mesh, material="gas", region="gas_top",
                     xl=0, xh=xo * UM, yl=(-tox - dg) * UM, yh=0)
    ds.add_2d_region(mesh=mesh, material="gas", region="gas_gate",
                     xl=xo * UM, xh=xh * UM, yl=(-tox - dg) * UM, yh=-tox * UM)
    ds.add_2d_region(mesh=mesh, material="gas", region="gas_bot",
                     xl=0, xh=xh * UM, yl=ym * UM, yh=(ym + dg) * UM)
    b = 1e-10
    ds.add_2d_contact(mesh=mesh, name="source", material="metal", region="sic",
                      xl=0, xh=xs * UM, yl=0, yh=0, bloat=b)
    ds.add_2d_contact(mesh=mesh, name="drain", material="metal", region="sic",
                      xl=0, xh=xh * UM, yl=ym * UM, yh=ym * UM, bloat=b)
    ds.add_2d_contact(mesh=mesh, name="gate", material="metal", region="oxide",
                      xl=xo * UM, xh=xh * UM, yl=-tox * UM, yh=-tox * UM, bloat=b)
    ds.add_2d_interface(mesh=mesh, name="ox_sic", region0="oxide", region1="sic",
                        xl=xo * UM, xh=xh * UM, yl=0, yh=0, bloat=b)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)

    x = ph.node_array(dev, "sic", "x") / UM
    y = ph.node_array(dev, "sic", "y") / UM
    don, acc = mosfet_doping(x, y, g, dop, npwell_scale)
    # interface length per node (cm): half of each neighbouring segment along y = 0 under the oxide
    on_intf = (np.abs(y) < 1e-9) & (x >= xo - 1e-9)
    intf = np.zeros_like(x)
    idx = np.where(on_intf)[0]
    order = idx[np.argsort(x[idx])]
    xi = x[order] * UM
    if len(xi) > 1:
        left = np.r_[0.0, np.diff(xi) / 2]
        right = np.r_[np.diff(xi) / 2, 0.0]
        intf[order] = left + right
    lam = cfg["material"].get("surf_lambda_um", 0.003)
    surf = np.where(x >= xo - 1e-9, np.exp(-np.maximum(y, 0) / lam), 0.0)
    nodes = len(x) + len(ph.node_array(dev, "oxide", "x"))
    info = dict(geometry=g, n_nodes=int(nodes), width_cm=xh * UM)   # half-cell width, for A/cm -> A/cm^2
    return dict(semis={"sic": (don, acc)}, ohmic=[("sic", "source"), ("sic", "drain")],
                oxides=["oxide"], gates=[("oxide", "gate")], interfaces=["ox_sic"],
                intf_len=intf, surf=surf, info=info)


# ------------------------------------------------- physics set-up + equilibrium
POISSON_TOL = dict(absolute_error=1e-9, relative_error=1e-10, maximum_iterations=60)


def setup_physics(dev, layout, mat: SiC4H, T, ii=True, interface=None, gate: Gate | None = None,
                  ox: SiO2 | None = None, circuit=None):
    """Parameters, doping, initial guess and the Poisson-only equations.

    circuit: optional {contact: circuit node name}, e.g. {"drain": "drain_bias"}.
    """
    gate = gate or Gate()
    ox = ox or SiO2()
    circuit = circuit or {}
    has_intf = bool(layout["interfaces"])
    d = None
    for reg, (don, acc) in layout["semis"].items():
        d = ph.set_sic_parameters(dev, reg, mat, T, interface)
        n = len(don)
        ph.set_node_array(dev, reg, "Donors", don)
        ph.set_node_array(dev, reg, "Acceptors", acc)
        il = layout["intf_len"] if layout["intf_len"] is not None else np.zeros(n)
        sf = layout["surf"] if layout["surf"] is not None else np.zeros(n)
        ph.set_node_array(dev, reg, "IntfLen", il)
        ph.set_node_array(dev, reg, "SurfFactor", sf)
        psi = ph.neutral_potential(don, acc, d, ii=ii)
        ph.set_node_array(dev, reg, "Psi_eq", psi)
        ph.solution(dev, reg, "Potential")
        ph.set_node_array(dev, reg, "Potential", psi)
        ph.create_sic_potential_only(dev, reg, ii=ii, interface=has_intf)
    for reg in layout["oxides"]:
        ph.set_oxide_parameters(dev, reg, ox)
        ph.create_oxide(dev, reg)
    for reg, c in layout["ohmic"]:
        ph.create_ohmic_contact(dev, reg, c, dd=False, circuit_node=circuit.get(c))
    for reg, c in layout["gates"]:
        ph.create_gate_contact(dev, reg, c, mat, gate, T)
    for itf in layout["interfaces"]:
        ph.create_oxide_interface(dev, itf)
    return d


def equilibrate(dev, layout, ii=True, tol=None, holes=True, circuit=None):
    """Poisson-only solve, then the drift-diffusion solve at zero bias."""
    circuit = circuit or {}
    solve_dc(dev, POISSON_TOL)                       # Poisson with equilibrium carriers
    has_intf = bool(layout["interfaces"])
    for reg in layout["semis"]:
        ph.create_sic_drift_diffusion(dev, reg, ii=ii, interface=has_intf, holes=holes)
    for reg, c in layout["ohmic"]:
        ph.create_ohmic_contact(dev, reg, c, dd=True, holes=holes, circuit_node=circuit.get(c))
    solve_dc(dev, tol)                               # drift-diffusion at equilibrium
