"""25. Adjoint sensitivity on the 1D SiC bar of step 05: dI/d(mu) and dI/d(ND) from one transposed
solve, checked against the formula I = q ND mu V / L and against re-solving.

  python steps/25_adjoint_resistor.py

The right contact is driven by a circuit voltage source V1, so the current "V1.I" is an unknown of the
Newton system and dI/dtheta = lam^T dR/dtheta with J^T lam = e_I.
"""
import contextlib
import io
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import devsim as ds
from sicdev.params import Q, EPS0, SiC4H
from sicdev.adjoint import adjoint_gradient

dev, reg = "bar", "sic"
L, ND, MU, VAPP = 10e-4, 1e16, 900.0, 1.0
d = SiC4H().derived(300.0)


def quiet_solve():
    with contextlib.redirect_stdout(io.StringIO()):
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)


with contextlib.redirect_stdout(io.StringIO()):
    ds.create_1d_mesh(mesh="m")
    ds.add_1d_mesh_line(mesh="m", pos=0.0, ps=0.1e-4, tag="left")
    ds.add_1d_mesh_line(mesh="m", pos=L, ps=0.1e-4, tag="right")
    ds.add_1d_region(mesh="m", material="SiC", region=reg, tag1="left", tag2="right")
    ds.add_1d_contact(mesh="m", name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh="m", name="right", tag="right", material="metal")
    ds.finalize_mesh(mesh="m")
    ds.create_device(mesh="m", device=dev)
for name, value in dict(Permittivity=9.7 * EPS0, ElectronCharge=Q, V_t=d["Vt"], n_i=d["ni"], ND=ND, mu_n=MU).items():
    ds.set_parameter(device=dev, region=reg, name=name, value=value)
ds.set_parameter(device=dev, name="left_bias", value=0.0)
n_nodes = len(ds.get_node_model_values(device=dev, region=reg, name="x"))
for name, init in (("Potential", d["Vt"] * math.log(ND / d["ni"])), ("Electrons", ND)):
    ds.node_solution(device=dev, region=reg, name=name)
    ds.edge_from_node_model(device=dev, region=reg, node_model=name)
    ds.set_node_values(device=dev, region=reg, name=name, values=[init] * n_nodes)


def nm(name, expr, *vs):
    ds.node_model(device=dev, region=reg, name=name, equation=expr)
    for v in vs:
        ds.node_model(device=dev, region=reg, name=f"{name}:{v}", equation=f"diff({expr}, {v})")


def em(name, expr, *vs):
    ds.edge_model(device=dev, region=reg, name=name, equation=expr)
    for v in vs:
        for e in ("n0", "n1"):
            ds.edge_model(device=dev, region=reg, name=f"{name}:{v}@{e}", equation=f"diff({expr}, {v}@{e})")


nm("SpaceCharge", "-ElectronCharge*(ND - Electrons)", "Electrons")
em("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength", "Potential")
em("DField", "Permittivity*ElectricField", "Potential")
ds.equation(device=dev, region=reg, name="PotentialEquation", variable_name="Potential",
            node_model="SpaceCharge", edge_model="DField", variable_update="log_damp")
em("vdiff", "(Potential@n0 - Potential@n1)/V_t", "Potential")
ds.edge_model(device=dev, region=reg, name="Bern01", equation="B(vdiff)")
ds.edge_model(device=dev, region=reg, name="Bern01:Potential@n0", equation="dBdx(vdiff)*vdiff:Potential@n0")
ds.edge_model(device=dev, region=reg, name="Bern01:Potential@n1", equation="-Bern01:Potential@n0")
em("ElectronCurrent",
   "ElectronCharge*mu_n*EdgeInverseLength*V_t*(Electrons@n1*Bern01 + Electrons@n1*vdiff - Electrons@n0*Bern01)",
   "Electrons", "Potential")
ds.equation(device=dev, region=reg, name="ElectronContinuityEquation", variable_name="Electrons",
            edge_model="ElectronCurrent", variable_update="positive")

# contacts: the boundary values use the parameters ND and n_i, so dR/dND also sees the contacts
ds.circuit_element(name="V1", n1="right_bias", n2=0, value=0.0)
for c in ("left", "right"):
    kw = dict(circuit_node="right_bias") if c == "right" else {}
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_phi", equation=f"Potential - V_t*log(ND/n_i) - {c}_bias")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_phi:Potential", equation="1")
    if c == "right":
        ds.contact_node_model(device=dev, contact=c, name="right_phi:right_bias", equation="-1")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_n", equation="Electrons - ND")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_n:Electrons", equation="1")
    ds.contact_equation(device=dev, contact=c, name="PotentialEquation", node_model=f"{c}_phi",
                        edge_charge_model="DField", **kw)
    ds.contact_equation(device=dev, contact=c, name="ElectronContinuityEquation", node_model=f"{c}_n",
                        edge_current_model="ElectronCurrent", **kw)
quiet_solve()
for i in range(1, 11):
    ds.circuit_alter(name="V1", value=VAPP * i / 10)
    quiet_solve()


def current():
    return ds.get_circuit_node_value(solution="dcop", node="V1.I")


I0 = current()
Ia = Q * ND * MU * VAPP / L
print(f"V = {VAPP} V: circuit current V1.I = {I0:.6e} A/cm^2, q ND mu V / L = {Ia:.6e} "
      f"(V1.I has the opposite sign of the current in the bar)")


def setter(name):
    return lambda v: ds.set_parameter(device=dev, region=reg, name=name, value=v)


params = {"mu_n": (setter("mu_n"), MU, MU * 1e-6), "ND": (setter("ND"), ND, ND * 1e-6)}
t0 = time.time()
grad, lam = adjoint_gradient("V1.I", params)
t_adj = time.time() - t0
print(f"\nadjoint: one transposed solve for both parameters, {t_adj * 1e3:.1f} ms, "
      f"{len(lam)} unknowns (Potential, Electrons, V1.I, right_bias)")


def resolve_fd(name, v0, h):
    """Reference: change the parameter, re-solve, central difference."""
    out = []
    for v in (v0 + h, v0 - h):
        setter(name)(v)
        quiet_solve()
        out.append(current())
    setter(name)(v0)
    quiet_solve()
    return (out[0] - out[1]) / (2 * h)


t0 = time.time()
fd = {k: resolve_fd(k, v0, v0 * 1e-4) for k, (_, v0, _) in params.items()}
t_fd = time.time() - t0
exact = {"mu_n": -Ia / MU, "ND": -Ia / ND}          # V1.I = -I_device here
print(f"{'theta':>6} {'adjoint dI/dtheta':>18} {'re-solve FD':>14} {'formula':>14} {'adj vs formula':>15}")
for k in params:
    print(f"{k:>6} {grad[k]:18.8e} {fd[k]:14.8e} {exact[k]:14.8e} {abs(grad[k] / exact[k] - 1):15.1e}")
print(f"time: adjoint {t_adj * 1e3:.1f} ms (both parameters), re-solve FD {t_fd * 1e3:.1f} ms (4 nonlinear solves)")
