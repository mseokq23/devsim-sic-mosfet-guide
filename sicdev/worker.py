"""Run one MOSFET job in its own process:

  python -m sicdev.worker job.json result.json

job.json: {"run_id", "temperature_K", "wjfet_scale", "npwell_scale", "qit_eff_cm2", "mu_channel_scale"}
"""
from __future__ import annotations

import json
import sys

from .config import load_config
from .mosfet import run_mosfet
from .utils import save_json, env_info


def run_job(job: dict) -> dict:
    cfg = load_config(job["config"]) if job.get("config") else load_config()
    rec = run_mosfet(cfg, job["temperature_K"], job.get("wjfet_scale", 1.0), job.get("npwell_scale", 1.0),
                     job.get("qit_eff_cm2"), job.get("mu_channel_scale", 1.0), job.get("mesh_scale"),
                     job.get("extended"), job.get("output_curve", True))
    return {**job, **rec, "env": env_info()}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        save_json(run_job(json.load(f)), sys.argv[2])
