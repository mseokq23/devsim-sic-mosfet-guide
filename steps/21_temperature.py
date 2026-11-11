"""21. 300 K vs 423 K: run the same protocol at 423 K and compare curves, features and where Ron goes.

  python steps/21_temperature.py      -> results/base_T423.json (needs results/base_T300.json from step 19)
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from, gate_from
from sicdev import mosfet, physics as ph, solver
from sicdev.mosfet import run_mosfet
from sicdev.params import Q, EPS0, gate_offset, ionized_fraction
from sicdev.utils import save_json, load_json
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "21"
cfg = load_config()
mat, gate = material_from(cfg), gate_from(cfg)
r300 = load_json(ROOT / "results" / "base_T300.json")
r423 = run_mosfet(cfg, 423.0)
save_json(r423, ROOT / "results" / "base_T423.json")
print(f"423 K run: converged {r423['converged']}, {r423['n_solves']} solves, {r423['runtime_s']:.1f} s\n")

KEYS = [("vth_V", "Vth (V)"), ("ss_mV_dec", "SS (mV/dec)"), ("gm_max_S_per_cm", "gm,max (S/cm)"),
        ("ion_A_per_cm", "Ion (A/cm)"), ("ron_mohm_cm2", "Ron,sp (mOhm cm2)"), ("id_vds_req_A_per_cm", "ID(2 V) (A/cm)")]
print(f"{'':18} {'300 K':>10} {'423 K':>10} {'change':>10}")
for k, name in KEYS:
    a, b = r300["features"][k], r423["features"][k]
    ch = f"{b - a:+.3f} V" if k == "vth_V" else f"{100 * (b / a - 1):+.1f} %"
    print(f"{name:18} {a:10.4g} {b:10.4g} {ch:>10}")

print("\nwhat changes with temperature (model inputs):")
d3, d4 = mat.derived(300.0), mat.derived(423.0)
print(f"  Vt = kT/q                    {d3['Vt'] * 1e3:7.2f} -> {d4['Vt'] * 1e3:7.2f} mV, "
      f"ideal SS = ln10 kT/q: {math.log(10) * d3['Vt'] * 1e3:.1f} -> {math.log(10) * d4['Vt'] * 1e3:.1f} mV/dec")
print(f"  ni                           {d3['ni']:.2e} -> {d4['ni']:.2e} cm^-3")
print(f"  P-well Al ionized (neutral)  {100 * ionized_fraction(1e17, 300, mat, 'acceptor'):.1f} -> "
      f"{100 * ionized_fraction(1e17, 423, mat, 'acceptor'):.1f} %")
print(f"  surface mobility mu_surf     {d3['mu_surf']:.1f} -> {d4['mu_surf']:.1f} cm2/Vs  (~ T^+1)")


def mub(N, d):
    return d["mun_min"] + (d["mun_max"] - d["mun_min"]) / (1 + (N / d["mun_Nref"]) ** d["mun_delta"])


print(f"  bulk mobility JFET 2e16      {mub(2e16, d3):.0f} -> {mub(2e16, d4):.0f} cm2/Vs")
print(f"  bulk mobility drift 1e16     {mub(1e16, d3):.0f} -> {mub(1e16, d4):.0f} cm2/Vs")

# threshold voltage terms from the long-channel MOS formula (net P-well 1e17 - 1e16 = 9e16, Qf + Qit = 0)
cox = 3.9 * EPS0 / (cfg["mosfet"]["geometry"]["tox_um"] * 1e-4)
print("\nlong-channel MOS formula, NA = 9e16 (net), Qf + Qit = 0:")
print(f"  {'':20} {'phi_m - phi_i':>13} {'psi_B':>8} {'2|psi_B|':>9} {'Qd/Cox':>8} {'Vth':>7}")
vh = {}
for T in (300.0, 423.0):
    d = mat.derived(T)
    NA = 9e16
    a = mat.gA / d["p1A"]
    p0 = (-1 + math.sqrt(1 + 4 * a * NA)) / (2 * a)
    psib = -d["Vt"] * math.log(p0 / d["ni"])
    qd = math.sqrt(2 * mat.eps_r * EPS0 * Q * NA * 2 * abs(psib)) / cox
    vh[T] = gate_offset(mat, gate, T) + psib + 2 * abs(psib) + qd
    print(f"  {T:5.0f} K {'':12} {gate_offset(mat, gate, T):+13.3f} {psib:+8.3f} {2 * abs(psib):9.3f} {qd:8.3f} "
          f"{vh[T]:7.3f}")
print(f"  change {vh[423.0] - vh[300.0]:+.3f} V (formula) vs {r423['features']['vth_V'] - r300['features']['vth_V']:+.3f} V "
      f"(simulation, ID = 1e-4 A/cm)")


def ron_split(T):
    """phi_n drops along the current path at VGS = 18 V, VDS = 0.1 V (fractions of VDS)."""
    b, tol = cfg["bias"], cfg["numerics"]["tol"]
    mosfet.build(cfg, T)
    solver.ramp("m", "drain", b["vds_lin"], 0.05, tol=tol)
    for stop, step in b["vgs_segments"]:
        solver.ramp("m", "gate", stop, step, tol=tol)
    solver.ramp("m", "gate", b["vgs_on"], 0.5, tol=tol)
    d = mat.derived(T)
    x = ph.node_array("m", "sic", "x") * 1e4
    y = ph.node_array("m", "sic", "y") * 1e4
    pf = ph.node_array("m", "sic", "Potential") - d["Vt"] * np.log(ph.node_array("m", "sic", "Electrons") / d["ni"])
    at = lambda xq, yq: pf[int(np.argmin((x - xq) ** 2 + (y - yq) ** 2))]
    v = [0.0, at(1.80, 0.0), at(2.30, 0.0), at(3.45, 1.0), b["vds_lin"]]
    return np.diff(v) / b["vds_lin"]


names = ["source", "channel", "JFET", "drift + substrate"]
split = {T: ron_split(T) for T in (300.0, 423.0)}
print("\nshare of VDS = 0.1 V at VGS = 18 V (quasi-Fermi potential drop), and the same in mOhm cm^2:")
for T in (300.0, 423.0):
    ron = (r300 if T == 300 else r423)["features"]["ron_mohm_cm2"]
    print(f"  {T:.0f} K: " + ", ".join(f"{n} {100 * s:.1f} % ({s * ron:.2f})" for n, s in zip(names, split[T])))

fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
for r, T, ls in ((r300, 300, "-"), (r423, 423, "--")):
    t, o = r["curves"]["transfer"], r["curves"]["output"]
    ax[0].semilogy(t["vgs"], np.abs(t["id"]), ls, label=f"{T} K")
    ax[1].plot(t["vgs"], np.array(t["id"]) * 1e3, ls, label=f"{T} K")
    ax[2].plot(o["vds"], o["id"], ls, label=f"{T} K")
ax[0].set(xlim=(2, 8), ylim=(1e-12, 1e-1), xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (A/cm)", title="(a) transfer (log)")
ax[1].set(xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (mA/cm)", title="(b) transfer, V$_{DS}$ = 0.1 V")
ax[2].set(xlabel="V$_{DS}$ (V)", ylabel="I$_D$ (A/cm)", title="(c) output, V$_{GS}$ = 18 V")
for a in ax[:3]:
    a.legend()
left = np.zeros(2)
for i, n in enumerate(names):
    vals = np.array([split[T][i] * (r300 if T == 300 else r423)["features"]["ron_mohm_cm2"] for T in (300.0, 423.0)])
    ax[3].barh(["300 K", "423 K"], vals, left=left, label=n)
    left += vals
ax[3].set(xlabel="R$_{on,sp}$ (m$\\Omega$cm$^2$)", title="(d) Ron,sp split", xlim=(0, 4.4))
ax[3].legend(fontsize=7, loc="center right")
savefig(fig, IMG / "temperature.png")
