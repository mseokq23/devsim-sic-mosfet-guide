"""22. Change one process parameter at a time (-20, -10, +10, +20 %) at 300 and 423 K and see which
features move. Uses the batch runner (one process per run, 2 in parallel).

  python steps/22_one_at_a_time.py      -> configs/design_oat.csv, results/oat/runs/*.json, runs.csv
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.utils import load_json
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "22"
NOMINAL = dict(wjfet_scale=1.0, npwell_scale=1.0, qit_eff_cm2=-1.0e12, mu_channel_scale=1.0)
LEVELS = (-0.2, -0.1, 0.1, 0.2)
LABEL = dict(wjfet_scale="W_JFET", npwell_scale="N_pw", qit_eff_cm2="|Qit,eff|", mu_channel_scale="mu_surf")
design = ROOT / "configs" / "design_oat.csv"
rows = [dict(candidate_id="A_base", **NOMINAL)]
for v in NOMINAL:
    for lv in LEVELS:
        r = dict(NOMINAL)
        r[v] = NOMINAL[v] * (1 + lv)
        rows.append(dict(candidate_id=f"A_{v.split('_')[0]}_{lv * 100:+.0f}pct", **r))
with open(design, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["candidate_id", *NOMINAL])
    w.writeheader()
    w.writerows(rows)
print(f"{design.relative_to(ROOT)}: {len(rows)} designs x 2 temperatures")
subprocess.run([sys.executable, "-m", "sicdev.runner", str(design.relative_to(ROOT)), "--out", "results/oat",
                "--jobs", "2"], cwd=ROOT, check=True)

R = {p.stem: load_json(p) for p in (ROOT / "results" / "oat" / "runs").glob("*.json")}
FEATS = [("vth_V", "Vth", "V"), ("ss_mV_dec", "SS", "%"), ("gm_max_S_per_cm", "gm,max", "%"),
         ("ion_A_per_cm", "Ion", "%"), ("ron_mohm_cm2", "Ron,sp", "%"), ("id_vds_req_A_per_cm", "ID(2 V)", "%")]
table = {}
for T in (300, 423):
    base = R[f"A_base_T{T}"]["features"]
    print(f"\n{T} K, change of each feature when one parameter goes +20 % (in brackets: -20 %)")
    print(f"{'':10}" + "".join(f"{n:>20}" for _, n, _ in FEATS))
    for v in NOMINAL:
        cells = []
        for k, n, unit in FEATS:
            vals = []
            for lv in (0.2, -0.2):
                f = R[f"A_{v.split('_')[0]}_{lv * 100:+.0f}pct_T{T}"]["features"]
                vals.append(f[k] - base[k] if unit == "V" else 100 * (f[k] / base[k] - 1))
            table[(T, v, k)] = vals[0]
            cells.append(f"{vals[0]:+.3f} ({vals[1]:+.3f}) V" if unit == "V" else f"{vals[0]:+.1f} ({vals[1]:+.1f}) %")
        print(f"{LABEL[v]:10}" + "".join(f"{c:>20}" for c in cells))

fig, ax = plt.subplots(2, 4, figsize=(15, 6.2))
cmap = plt.get_cmap("coolwarm")
for j, v in enumerate(NOMINAL):
    for lv in (-0.2, -0.1, 0.0, 0.1, 0.2):
        key = "A_base" if lv == 0 else f"A_{v.split('_')[0]}_{lv * 100:+.0f}pct"
        c = cmap(0.5 + lv * 2.2)
        for T, ls in ((300, "-"), (423, "--")):
            r = R[f"{key}_T{T}"]
            t, o = r["curves"]["transfer"], r["curves"]["output"]
            lab = f"{lv * 100:+.0f} %" if T == 300 else None
            ax[0, j].plot(t["vgs"], np.array(t["id"]) * 1e3, ls, color=c, lw=1, label=lab)
            ax[1, j].plot(o["vds"], o["id"], ls, color=c, lw=1, label=lab)
    ax[0, j].set(title=f"{LABEL[v]} (solid 300 K, dashed 423 K)", xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (mA/cm)")
    ax[1, j].set(xlabel="V$_{DS}$ (V)", ylabel="I$_D$ (A/cm)")
    ax[0, j].legend(fontsize=7)
savefig(fig, IMG / "one_at_a_time.png")
