"""23. Many runs: a Sobol design of 16 devices x 2 temperatures with the batch runner, then the same
command again to show that finished runs are skipped.

  python steps/23_doe.py      -> configs/design_doe.csv, results/doe/runs/*.json, results/doe/runs.csv
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import qmc
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "23"
BOUNDS = dict(wjfet_scale=(0.8, 1.2), npwell_scale=(0.8, 1.2), qit_eff_cm2=(-1.5e12, -0.5e12),
              mu_channel_scale=(0.8, 1.2))
names = list(BOUNDS)
u = qmc.Sobol(d=len(names), scramble=True, seed=7).random_base2(4)   # 2^4 = 16 points in [0, 1)^4
# (the sic-mosfet-devsim pool used seed 2026; another seed gives points that are not in that data)
lo, hi = np.array([BOUNDS[k][0] for k in names]), np.array([BOUNDS[k][1] for k in names])
x = lo + u * (hi - lo)
design = ROOT / "configs" / "design_doe.csv"
with open(design, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate_id", *names])
    for i, row in enumerate(x):
        w.writerow([f"D{i:02d}", *[f"{v:.6g}" for v in row]])
print(f"{design.relative_to(ROOT)}: 16 Sobol points, discrepancy {qmc.discrepancy(u):.4f} "
      f"(random 16 points: {qmc.discrepancy(np.random.default_rng(0).random((16, 4))):.4f})")

cmd = [sys.executable, "-m", "sicdev.runner", str(design.relative_to(ROOT)), "--out", "results/doe", "--jobs", "2"]
print("\n$ python -m sicdev.runner configs/design_doe.csv --out results/doe --jobs 2")
subprocess.run(cmd, cwd=ROOT, check=True)
print("\n$ python -m sicdev.runner configs/design_doe.csv --out results/doe --jobs 2   (again)")
subprocess.run(cmd, cwd=ROOT, check=True)

with open(ROOT / "results" / "doe" / "runs.csv") as f:
    runs = list(csv.DictReader(f))
print(f"\nruns.csv: {len(runs)} rows, {len(runs[0])} columns, all converged: "
      f"{all(r['converged'] == 'True' for r in runs)}")
rt = np.array([float(r["runtime_s"]) for r in runs])
print(f"runtime per run: median {np.median(rt):.0f} s, max {rt.max():.0f} s")

fig, ax = plt.subplots(2, 4, figsize=(15, 6))
for i, (k, (a, b)) in enumerate(BOUNDS.items()):
    for T, mk in ((300, "o"), (423, "s")):
        sel = [r for r in runs if float(r["temperature_K"]) == T]
        xv = [float(r[k]) for r in sel]
        ax[0, i].plot(xv, [float(r["vth_V"]) for r in sel], mk, mfc="none", label=f"{T} K")
        ax[1, i].plot(xv, [float(r["ron_mohm_cm2"]) for r in sel], mk, mfc="none", label=f"{T} K")
    ax[1, i].set(xlabel=k)
    ax[0, i].legend(fontsize=7)
ax[0, 0].set_ylabel("Vth (V)")
ax[1, 0].set_ylabel("Ron,sp (m$\\Omega$cm$^2$)")
savefig(fig, IMG / "doe_features.png")
