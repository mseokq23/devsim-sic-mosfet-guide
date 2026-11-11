"""26. Adjoint sensitivities of the SiC MOSFET drain current (300 K, VDS = 0.1 V):
d ID / d(Qit,eff), d ID / d(N_pw), d ID / d(mu_surf), d ID / d VGS from one transposed solve per bias point,
checked against re-solving, then along the whole transfer curve.

  python steps/26_adjoint_mosfet.py

The drain is tied to a circuit voltage source (Vdrain), so the drain current is the unknown "Vdrain.I".
"""
import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import devsim as ds
from sicdev.config import load_config, material_from
from sicdev import mosfet, physics as ph, solver
from sicdev.devices import mosfet_doping
from sicdev.params import Q, EPS0
from sicdev.adjoint import adjoint_gradient
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "26"
cfg = load_config()
b, tol = cfg["bias"], cfg["numerics"]["tol"]
T = 300.0
d = material_from(cfg).derived(T)
COX = 3.9 * EPS0 / (cfg["mosfet"]["geometry"]["tox_um"] * 1e-4)

# reference: the usual model (drain bias as a device parameter) at VGS = 6 V
mosfet.build(cfg, T)
solver.ramp("m", "drain", b["vds_lin"], 0.05, tol=tol)
solver.ramp("m", "gate", 6.0, 0.25, tol=tol)
i_ref = ph.contact_current("m", "drain")

# the same device with the drain on a circuit voltage source
ds.circuit_element(name="Vdrain", n1="drain_bias", n2=0, value=0.0)
lay = mosfet.build(cfg, T, circuit={"drain": "drain_bias"})
for v in (0.05, b["vds_lin"]):
    ds.circuit_alter(name="Vdrain", value=v)
    solver.solve_dc("m", tol)


def drain_current():
    return -ds.get_circuit_node_value(solution="dcop", node="Vdrain.I")


solver.ramp("m", "gate", 6.0, 0.25, tol=tol)
print(f"VGS = 6 V, VDS = 0.1 V: ID = {i_ref:.10e} A/cm (parameter drain), {drain_current():.10e} A/cm (circuit drain)")

# --- parameters: how to change each one without solving
g = lay["info"]["geometry"]
x = ph.node_array("m", "sic", "x") * 1e4
y = ph.node_array("m", "sic", "y") * 1e4
dop = cfg["mosfet"]["doping"]


def set_sigma(v):
    ds.set_parameter(device="m", region="sic", name="SigmaFixed", value=v)


def set_mu_surf(v):
    ds.set_parameter(device="m", region="sic", name="mu_surf", value=v)


def set_gate(v):
    ds.set_parameter(device="m", name="gate_bias", value=v)


def set_npwell(s):
    """P-well peak doping scale: new acceptor profile and the matching ohmic boundary value Psi_eq."""
    don, acc = mosfet_doping(x, y, g, dop, s)
    ph.set_node_array("m", "sic", "Acceptors", acc)
    ph.set_node_array("m", "sic", "Psi_eq", ph.neutral_potential(don, acc, d, ii=True))


sigma0 = ds.get_parameter(device="m", region="sic", name="SigmaFixed")
mus0 = ds.get_parameter(device="m", region="sic", name="mu_surf")


def params(vg):
    return {"Qit": (set_sigma, sigma0, Q * 1e9),           # sheet charge q*(Qf + Qit), step = 1e9 cm^-2
            "N_pw": (set_npwell, 1.0, 1e-4),
            "mu_surf": (set_mu_surf, mus0, mus0 * 1e-6),
            "VGS": (set_gate, vg, 1e-6)}


def snapshot():
    s = {v: ph.node_array("m", "sic", v) for v in ("Potential", "Electrons")}
    s["ox"] = ph.node_array("m", "oxide", "Potential")
    s["c"] = {n: ds.get_circuit_node_value(solution="dcop", node=n) for n in ("drain_bias", "Vdrain.I")}
    return s


def restore(s):
    for v in ("Potential", "Electrons"):
        ds.set_node_values(device="m", region="sic", name=v, values=list(s[v]))
    ds.set_node_values(device="m", region="oxide", name="Potential", values=list(s["ox"]))
    for n, v in s["c"].items():
        ds.set_circuit_node_value(solution="dcop", node=n, value=v)


def resolve_fd(setter, v0, h):
    """Reference: change theta by +-h, solve again (Newton), central difference."""
    s0, out = snapshot(), []
    for v in (v0 + h, v0 - h):
        setter(v)
        solver.solve_dc("m", dict(tol, relative_error=1e-10, maximum_iterations=60))
        out.append(drain_current())
        restore(s0)
    setter(v0)
    return (out[0] - out[1]) / (2 * h)


def per_unit(name, g_):
    """Gradient in reporting units: Qit per 1e12 cm^-2, N_pw per +100 % (scale 1 -> 2), mu_surf per cm2/Vs, VGS per V."""
    return g_ * Q * 1e12 if name == "Qit" else g_


print("\nadjoint vs re-solve (central difference) at VDS = 0.1 V")
print(f"{'VGS':>5} {'ID (A/cm)':>11} {'param':>8} {'adjoint':>14} {'re-solve FD':>14} {'rel. diff':>9}")
FD_H = {"Qit": Q * 1e8, "N_pw": 1e-4, "mu_surf": 1e-3 * mus0, "VGS": 1e-4}   # small: ID is exponential below Vth
t_adj, t_fd, n_pts = 0.0, 0.0, 0
for vg in (3.5, 4.0, 6.0, 10.0, 18.0):
    solver.ramp("m", "gate", vg, 0.25, tol=tol)
    solver.solve_dc("m", dict(tol, relative_error=1e-10, maximum_iterations=60))   # tight solve before gradients
    i0 = drain_current()
    t0 = time.time()
    grad, _ = adjoint_gradient("Vdrain.I", params(vg))
    t_adj += time.time() - t0
    t0 = time.time()
    fd = {k: resolve_fd(p[0], p[1], FD_H[k]) for k, p in params(vg).items()}
    t_fd += time.time() - t0
    n_pts += 1
    for k in grad:
        ga, gf = -per_unit(k, grad[k]), per_unit(k, fd[k])
        print(f"{vg:5.1f} {i0:11.3e} {k:>8} {ga:14.6e} {gf:14.6e} {abs(ga / gf - 1):9.1e}")
print(f"time per bias point: adjoint {1e3 * t_adj / n_pts:.0f} ms (4 parameters, 1 factorization), "
      f"re-solve FD {1e3 * t_fd / n_pts:.0f} ms (8 Newton solves)")
print("units: Qit per 1e12 cm^-2 (A/cm per 1e12), N_pw per unit scale, mu_surf per cm2/Vs, VGS per V (= gm)")

# --- along the transfer curve
solver.ramp("m", "gate", 3.0, 0.25, tol=tol)
VG, S = [], {k: [] for k in ("Qit", "N_pw", "mu_surf", "VGS")}
ID = []
for vg in np.arange(3.0, 20.0001, 0.25):
    solver.ramp("m", "gate", float(vg), 0.25, tol=tol)
    i0 = drain_current()
    grad, _ = adjoint_gradient("Vdrain.I", params(float(vg)))
    VG.append(vg)
    ID.append(i0)
    for k in S:
        S[k].append(-per_unit(k, grad[k]))
VG, ID = np.array(VG), np.array(ID)
S = {k: np.array(v) for k, v in S.items()}
ok = ID > 1e-9
# the same changes as in step 22 (+10 %), written as the gate-voltage shift that gives the same dID
dQ, dN, dM = -0.1, 0.1, 0.1 * mus0            # |Qit,eff| +10 % (Qit -1e11), N_pw +10 %, mu_surf +10 %
eq = {"|Qit,eff| +10 %": dQ * S["Qit"] / S["VGS"], "N_pw +10 %": dN * S["N_pw"] / S["VGS"],
      "mu_surf +10 %": dM * S["mu_surf"] / S["VGS"]}
print(f"\nalong the transfer curve (VGS 3..20 V, ID > 1e-9 A/cm, {ok.sum()} points), "
      f"equivalent VGS shift = (dID/dtheta * dtheta) / (dID/dVGS):")
for k, v in eq.items():
    print(f"  {k:16}: {v[ok].min():+.4f} .. {v[ok].max():+.4f} V")
print(f"  (q/Cox * 1e11 cm^-2 = {Q / COX * 1e11:.4f} V)")
print("  -> a change of Qit,eff is exactly a shift of VGS at every bias; N_pw and mu_surf are not")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, v in eq.items():
    th = {"|Qit,eff| +10 %": dQ * S["Qit"], "N_pw +10 %": dN * S["N_pw"], "mu_surf +10 %": dM * S["mu_surf"]}[k]
    ax[0].plot(VG, th / (ID * np.log(10)), label=k)
    ax[1].plot(VG[ok], v[ok], label=k)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set(xlabel="V$_{GS}$ (V)", ylabel="$\\Delta$ log$_{10}$ I$_D$ (decades, linearised)",
          title="(a) change of log I$_D$, 300 K, V$_{DS}$ = 0.1 V")
ax[1].axhline(-Q / COX * 1e11, color="gray", ls="--", lw=0.8)
ax[1].text(10.5, -Q / COX * 1e11 + 0.03, "-q/C$_{ox}$ x 1e11 cm$^{-2}$ = -0.232 V", color="gray", fontsize=8)
ax[1].set(xlabel="V$_{GS}$ (V)", ylabel="equivalent V$_{GS}$ shift (V)",
          title="(b) the same change written as a V$_{GS}$ shift")
for a in ax:
    a.legend(fontsize=8)
savefig(fig, IMG / "adjoint_sensitivity.png")
