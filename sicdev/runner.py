"""Run a design table: every (row, temperature) in a separate worker process.

  python -m sicdev.runner design.csv --out results/doe --jobs 2 [--temps 300 423]

* one JSON per run in <out>/runs/<run_id>.json, written as soon as the run ends
* runs that already have a JSON are skipped -> the same command continues an interrupted batch
* <out>/runs.csv: one row per run (design values + features), rebuilt at the end
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN_VARS = ("wjfet_scale", "npwell_scale", "qit_eff_cm2", "mu_channel_scale")


def read_design(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in DESIGN_VARS:
            if k in r:
                r[k] = float(r[k])
    return rows


def run_one(job, out_json):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "sicdev.worker", f.name, str(out_json)],
                       env=env, capture_output=True, text=True)
    os.unlink(f.name)
    if p.returncode != 0:
        return job["run_id"], False, f"worker error: {p.stderr.strip()[-300:]}"
    with open(out_json) as g:
        r = json.load(g)
    msg = f"{time.time() - t0:5.1f} s, converged={r['converged']}"
    if r["converged"]:
        msg += f", Vth={r['features']['vth_V']:.3f} V, Ron={r['features']['ron_mohm_cm2']:.3f} mOhm cm2"
    return job["run_id"], r["converged"], msg


def merge(out_dir):
    rows = []
    for p in sorted((out_dir / "runs").glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        row = {k: r.get(k) for k in ("run_id", "candidate_id", "temperature_K", *DESIGN_VARS,
                                     "converged", "runtime_s", "n_solves", "n_fail")}
        row.update(r.get("features") or {})
        rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--temps", type=float, nargs="+", default=[300.0, 423.0])
    a = ap.parse_args()
    out_dir = Path(a.out)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    todo, skipped = [], 0
    for row in read_design(a.design):
        for T in a.temps:
            run_id = f"{row['candidate_id']}_T{T:.0f}"
            out_json = out_dir / "runs" / f"{run_id}.json"
            if out_json.exists():
                skipped += 1
                continue
            job = {k: row[k] for k in ("candidate_id", *DESIGN_VARS) if k in row}
            job.update(run_id=run_id, temperature_K=T)
            todo.append((job, out_json))
    print(f"{len(todo)} runs to do, {skipped} already done, {a.jobs} parallel", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as pool:
        for run_id, ok, msg in pool.map(lambda jo: run_one(*jo), todo):
            print(f"  {run_id}: {msg}", flush=True)
    n = merge(out_dir)
    print(f"done in {time.time() - t0:.0f} s, {n} runs in {out_dir / 'runs.csv'}")


if __name__ == "__main__":
    main()
