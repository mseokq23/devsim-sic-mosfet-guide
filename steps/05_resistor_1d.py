"""05. First DEVSIM device: a 1D n-type 4H-SiC bar (resistor), written with the raw DEVSIM API.

  python steps/05_resistor_1d.py

Electron-only drift-diffusion: Poisson + electron continuity (SG current), two ohmic contacts.
Check: current density J = q * ND * mu * V / L (uniform bar, all donors ionized).
"""
import contextlib
import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import devsim as ds
from sicdev.params import Q, EPS0, SiC4H
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "05"
dev, reg = "bar", "sic"
L = 10e-4          # cm (10 um)
ND = 1e16          # cm^-3
MU = 900.0         # cm^2/Vs, constant here
d = SiC4H().derived(300.0)
VT, NI = d["Vt"], d["ni"]

# ---- 1. mesh: two lines, 0.1 um spacing, a contact at each end
with contextlib.redirect_stdout(io.StringIO()):          # DEVSIM prints the mesh it builds; hide it
    ds.create_1d_mesh(mesh="bar_mesh")
    ds.add_1d_mesh_line(mesh="bar_mesh", pos=0.0, ps=0.1e-4, tag="left")
    ds.add_1d_mesh_line(mesh="bar_mesh", pos=L, ps=0.1e-4, tag="right")
    ds.add_1d_region(mesh="bar_mesh", material="SiC", region=reg, tag1="left", tag2="right")
    ds.add_1d_contact(mesh="bar_mesh", name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh="bar_mesh", name="right", tag="right", material="metal")
    ds.finalize_mesh(mesh="bar_mesh")
    ds.create_device(mesh="bar_mesh", device=dev)

# ---- 2. parameters
for name, value in dict(Permittivity=9.7 * EPS0, ElectronCharge=Q, V_t=VT, n_i=NI, ND=ND, mu_n=MU).items():
    ds.set_parameter(device=dev, region=reg, name=name, value=value)
for c in ("left", "right"):
    ds.set_parameter(device=dev, name=f"{c}_bias", value=0.0)


def node_model(name, expr, *variables):
    ds.node_model(device=dev, region=reg, name=name, equation=expr)
    for v in variables:
        ds.node_model(device=dev, region=reg, name=f"{name}:{v}", equation=f"diff({expr}, {v})")


def edge_model(name, expr, *variables):
    ds.edge_model(device=dev, region=reg, name=name, equation=expr)
    for v in variables:
        for end in ("n0", "n1"):
            ds.edge_model(device=dev, region=reg, name=f"{name}:{v}@{end}", equation=f"diff({expr}, {v}@{end})")


# ---- 3. unknowns and initial guess (neutral n-type: n = ND, Potential = Vt ln(ND/ni))
psi0 = VT * math.log(ND / NI)
n_nodes = len(ds.get_node_model_values(device=dev, region=reg, name="x"))
for name, init in (("Potential", psi0), ("Electrons", ND)):
    ds.node_solution(device=dev, region=reg, name=name)
    ds.edge_from_node_model(device=dev, region=reg, node_model=name)
    ds.set_node_values(device=dev, region=reg, name=name, values=[init] * n_nodes)
print(f"nodes = {n_nodes}, Vt = {VT:.5f} V, ni = {NI:.3e} cm^-3, Potential(n-type) = {psi0:.4f} V")

# ---- 4. Poisson:  div(eps E) = rho = q (ND - n)
node_model("SpaceCharge", "-ElectronCharge*(ND - Electrons)", "Electrons")
edge_model("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength", "Potential")
edge_model("DField", "Permittivity*ElectricField", "Potential")
ds.equation(device=dev, region=reg, name="PotentialEquation", variable_name="Potential",
            node_model="SpaceCharge", edge_model="DField", variable_update="log_damp")

# ---- 5. electron continuity:  div(Jn) = 0,  Jn by Scharfetter-Gummel
edge_model("vdiff", "(Potential@n0 - Potential@n1)/V_t", "Potential")
ds.edge_model(device=dev, region=reg, name="Bern01", equation="B(vdiff)")
ds.edge_model(device=dev, region=reg, name="Bern01:Potential@n0", equation="dBdx(vdiff)*vdiff:Potential@n0")
ds.edge_model(device=dev, region=reg, name="Bern01:Potential@n1", equation="-Bern01:Potential@n0")
edge_model("ElectronCurrent",
           "ElectronCharge*mu_n*EdgeInverseLength*V_t*(Electrons@n1*Bern01 + Electrons@n1*vdiff - Electrons@n0*Bern01)",
           "Electrons", "Potential")
ds.equation(device=dev, region=reg, name="ElectronContinuityEquation", variable_name="Electrons",
            edge_model="ElectronCurrent", variable_update="positive")

# ---- 6. ohmic contacts: Potential = psi0 + V, n = ND
for c in ("left", "right"):
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_phi", equation=f"Potential - {psi0} - {c}_bias")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_phi:Potential", equation="1")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_n", equation="Electrons - ND")
    ds.contact_node_model(device=dev, contact=c, name=f"{c}_n:Electrons", equation="1")
    ds.contact_equation(device=dev, contact=c, name="PotentialEquation", node_model=f"{c}_phi",
                        edge_charge_model="DField")
    ds.contact_equation(device=dev, contact=c, name="ElectronContinuityEquation", node_model=f"{c}_n",
                        edge_current_model="ElectronCurrent")

# ---- 7. solve at 0 V, then sweep the right contact (Newton log shown for the 0.1 V step only)
ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)
V, J, Ja = [], [], []
for i in range(11):
    v = 0.1 * i
    ds.set_parameter(device=dev, name="right_bias", value=v)
    if i == 1:
        print("\n--- Newton iterations, right contact 0 -> 0.1 V ---")
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)
        print("--- end ---\n")
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)
    j = ds.get_contact_current(device=dev, contact="right", equation="ElectronContinuityEquation")
    V.append(v)
    J.append(j)
    Ja.append(Q * ND * MU * v / L)
print(f"{'V (V)':>6} {'J DEVSIM (A/cm2)':>17} {'J = qNDmuV/L':>14} {'rel diff':>9}")
for v, j, ja in zip(V, J, Ja):
    rd = abs(abs(j) - ja) / ja if ja else 0.0
    print(f"{v:6.1f} {j:17.6e} {ja:14.6e} {rd:9.1e}")
print(f"\nspecific resistance from the slope: {1e3 * V[-1] / abs(J[-1]):.4f} mOhm cm^2 "
      f"(rho*L = L/(q ND mu) = {1e3 * L / (Q * ND * MU):.4f})")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(V, [abs(j) for j in J], "o", label="DEVSIM")
ax[0].plot(V, Ja, "-", label="q ND mu V / L")
ax[0].set(xlabel="V (V)", ylabel="|J| (A/cm$^2$)", title="1D SiC bar, ND = 1e16, L = 10 um")
ax[0].legend()
x = [p * 1e4 for p in ds.get_node_model_values(device=dev, region=reg, name="x")]
ax[1].plot(x, ds.get_node_model_values(device=dev, region=reg, name="Potential"))
ax[1].set(xlabel="x (um)", ylabel="Potential (V)", title="Potential at V = 1 V")
savefig(fig, IMG / "resistor_iv.png")
