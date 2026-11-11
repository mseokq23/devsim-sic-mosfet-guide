"""20. Can we trust the numbers? Mesh study (coarse / medium / fine), double vs 128-bit precision,
and a repeated run in a new process. Each case runs in its own process (sicdev.worker).

  python steps/20_mesh_study.py      -> results/mesh_study/*.json
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.runner import run_one
from sicdev.utils import load_json
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "20"
OUT = ROOT / "results" / "mesh_study"
OUT.mkdir(parents=True, exist_ok=True)
cases = {
    "coarse": dict(mesh_scale=2.0),
    "medium": dict(mesh_scale=1.5),
    "fine": dict(mesh_scale=1.0),
    "medium_128bit": dict(mesh_scale=1.5, extended=True, output_curve=False),
}
t0 = time.time()
jobs = [(dict(run_id=k, temperature_K=300.0, **v), OUT / f"{k}.json") for k, v in cases.items()]
with ThreadPoolExecutor(2) as pool:
    for run_id, ok, msg in pool.map(lambda jo: run_one(*jo), jobs):
        print(f"  {run_id:14s} {msg}")
print(f"all cases {time.time() - t0:.0f} s (2 in parallel)\n")
R = {k: load_json(OUT / f"{k}.json") for k in cases}

KEYS = [("vth_V", "Vth (V)", "abs", 0.020), ("ss_mV_dec", "SS (mV/dec)", "rel", 0.05),
        ("gm_max_S_per_cm", "gm,max (S/cm)", "rel", 0.03), ("ion_A_per_cm", "Ion (A/cm)", "rel", 0.02),
        ("ron_mohm_cm2", "Ron,sp (mOhm cm2)", "rel", 0.02)]
print(f"{'':18} {'coarse':>10} {'medium':>10} {'fine':>10} {'medium-fine':>12} {'limit':>7} {'ok':>3}")
print(f"{'nodes':18} {R['coarse']['n_nodes']:10d} {R['medium']['n_nodes']:10d} {R['fine']['n_nodes']:10d}")
print(f"{'runtime (s)':18} {R['coarse']['runtime_s']:10.1f} {R['medium']['runtime_s']:10.1f} {R['fine']['runtime_s']:10.1f}")
print(f"{'solves':18} {R['coarse']['n_solves']:10d} {R['medium']['n_solves']:10d} {R['fine']['n_solves']:10d}")
print(f"{'failed steps':18} {R['coarse']['n_fail']:10d} {R['medium']['n_fail']:10d} {R['fine']['n_fail']:10d}")
reach = {k: max(R[k]["curves"]["output"]["vds"]) for k in ("coarse", "medium", "fine")}
print(f"{'output up to (V)':18} {reach['coarse']:10.4g} {reach['medium']:10.4g} {reach['fine']:10.4g}")
for k, name, kind, lim in KEYS:
    c, m, f = (R[x]["features"][k] for x in ("coarse", "medium", "fine"))
    dmf = (m - f) if kind == "abs" else (m - f) / f
    txt = f"{dmf * 1e3:+.2f} mV" if kind == "abs" else f"{100 * dmf:+.2f} %"
    lt = f"{lim * 1e3:.0f} mV" if kind == "abs" else f"{100 * lim:.0f} %"
    print(f"{name:18} {c:10.5g} {m:10.5g} {f:10.5g} {txt:>12} {lt:>7} {'yes' if abs(dmf) <= lim else 'NO':>3}")

xp, dp = R["medium_128bit"], R["medium"]
print("\ndouble vs 128-bit (medium mesh, transfer curve only):")
for k, name, kind, _ in KEYS[:4]:
    a, b_ = dp["features"][k], xp["features"][k]
    print(f"  {name:16} double {a:.10g}  128-bit {b_:.10g}  diff {a - b_:+.1e}")
print(f"  numerical floor max|ID + IS|: double {dp['features']['kcl_floor_A_per_cm']:.1e} A/cm, "
      f"128-bit {xp['features']['kcl_floor_A_per_cm']:.1e} A/cm")
print(f"  runtime: double {dp['runtime_s']:.0f} s for {dp['n_solves']} solves (transfer + output), "
      f"128-bit {xp['runtime_s']:.0f} s for {xp['n_solves']} solves (transfer only) -> "
      f"{(xp['runtime_s'] / xp['n_solves']) / (dp['runtime_s'] / dp['n_solves']):.1f}x per solve")

base = ROOT / "results" / "base_T300.json"
if base.exists():
    b0 = load_json(base)
    same = all(np.array_equal(b0["curves"][c][k], dp["curves"][c][k])
               for c, ks in (("transfer", ("vgs", "id", "is_")), ("output", ("vds", "id"))) for k in ks)
    print(f"\nrepeat: medium run vs results/base_T300.json (step 19, another process): "
          f"{'bit-for-bit identical' if same else 'DIFFERENT'}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for k, ls in (("coarse", ":"), ("medium", "-"), ("fine", "--")):
    t = R[k]["curves"]["transfer"]
    ax[0].plot(t["vgs"], np.array(t["id"]) * 1e3, ls, label=f"{k} ({R[k]['n_nodes']} nodes)")
ax[0].set(xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (mA/cm)", title="(a) mesh study, 300 K, V$_{DS}$ = 0.1 V")
ax[0].legend()
for r, lab, st in ((dp, "double", "-"), (xp, "128-bit", "--")):
    t = r["curves"]["transfer"]
    ax[1].semilogy(t["vgs"], np.abs(t["id"]), st, label=lab)
    ax[1].axhline(r["features"]["kcl_floor_A_per_cm"], ls=st, lw=0.7, color="gray")
ax[1].set(xlabel="V$_{GS}$ (V)", ylabel="|I$_D$| (A/cm)", ylim=(1e-30, 1e-1),
          title="(b) precision: current floor (gray lines)")
ax[1].legend()
savefig(fig, IMG / "mesh_precision.png")
