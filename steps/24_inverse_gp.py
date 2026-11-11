"""24. Inverse estimation with the paper's data: from I-V features back to W_JFET, N_pw and Qit,eff.

  git clone https://github.com/mseokq23/sic-mosfet-devsim ../sic-mosfet-devsim
  python steps/24_inverse_gp.py --src ../sic-mosfet-devsim

Train on the 512 pool devices, test on the 128 test devices (results/pool, results/test of that repo).
Measurement-like noise is added to the features (Vth 10 mV, SS and gm 2 %, currents and Ron 1 %,
log ID samples 2 %; samples below 1e-11 A/cm are clipped), then a Gaussian-process regression maps
features -> parameters. S1 = 300 K features (17), S2 = 300 K + 423 K features (34).
The 16 devices of step 23 are predicted with the same S2 model at the end.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "24"
TARGETS = ["wjfet_scale", "npwell_scale", "qit_eff_cm2"]
LO, HI = np.array([0.8, 0.8, -1.5e12]), np.array([1.2, 1.2, -0.5e12])
SCALAR = ["vth_V", "ss_mV_dec", "gm_max_S_per_cm", "ion_A_per_cm", "ron_mohm_cm2", "id_vds_req_A_per_cm"]
SAMPLES = [f"logid_vg{v}" for v in (4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20)]
LOG10 = {"gm_max_S_per_cm", "ion_A_per_cm", "ron_mohm_cm2", "id_vds_req_A_per_cm"}
NOISE_ABS = {"vth_V": 0.010}
NOISE_REL = {"ss_mV_dec": 0.02, "gm_max_S_per_cm": 0.02, "ion_A_per_cm": 0.01, "ron_mohm_cm2": 0.01,
             "id_vds_req_A_per_cm": 0.01, **{s: 0.02 for s in SAMPLES}}
FLOOR = np.log10(1e-11)


def wide_table(path):
    """One row per device (sorted by id): [17 features at 300 K, 17 at 423 K] and the true parameters."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["converged"].lower() != "true":
                continue
            rows.setdefault(r["candidate_id"], {})[round(float(r["temperature_K"]))] = r
    ids = sorted(i for i, d in rows.items() if 300 in d and 423 in d)
    cols = [(T, k) for T in (300, 423) for k in SCALAR + SAMPLES]
    X = np.array([[np.log10(max(float(rows[i][T][k]), 1e-30)) if k in LOG10 else float(rows[i][T][k])
                   for T, k in cols] for i in ids])
    Y = np.array([[float(rows[i][300][t]) for t in TARGETS] for i in ids])
    return ids, cols, X, Y


def add_noise(X, cols, seed):
    """One draw per column (in column order) from one generator; relative noise on log10 columns is
    added as log10(1 + eps)."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for j, (_, k) in enumerate(cols):
        z = rng.standard_normal(len(X))
        if k in NOISE_ABS:
            out[:, j] += NOISE_ABS[k] * z
            continue
        eps = np.clip(NOISE_REL[k] * z, -0.5, 0.5)
        out[:, j] = out[:, j] + np.log10(1 + eps) if (k in LOG10 or k in SAMPLES) else out[:, j] * (1 + eps)
        if k in SAMPLES:
            out[:, j] = np.maximum(out[:, j], FLOOR)
    return out


def gp_fit_predict(Xtr, Ytr, Xte):
    k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
    m = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, random_state=0))
    m.fit(Xtr, (Ytr - LO) / (HI - LO))
    return LO + m.predict(Xte) * (HI - LO)


ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT.parent / "sic-mosfet-devsim"), help="clone of sic-mosfet-devsim")
a = ap.parse_args()
src = Path(a.src)
_, cols, Xp, Yp = wide_table(src / "results" / "pool" / "runs.csv")
ids_t, _, Xt, Yt = wide_table(src / "results" / "test" / "runs.csv")
print(f"pool {len(Xp)} devices, test {len(Xt)} devices, {len(cols)} features per device (17 per temperature)")
Xp_n, Xt_n = add_noise(Xp, cols, 101), add_noise(Xt, cols, 202)
s1 = [j for j, (T, _) in enumerate(cols) if T == 300]
pred, err = {}, {}
for name, sel in (("S1 (300 K)", s1), ("S2 (300 + 423 K)", list(range(len(cols))))):
    P = gp_fit_predict(Xp_n[:, sel], Yp, Xt_n[:, sel])
    e = np.abs(P - Yt) / (HI - LO)
    pred[name], err[name] = P, e
    print(f"{name:17} MAE_norm mean {e.mean():.6f} | W_JFET {e[:, 0].mean():.4f}  N_pw {e[:, 1].mean():.4f}  "
          f"Qit {e[:, 2].mean():.4f}")
print("(MAE_norm = |estimate - true| / parameter range; guessing the range center gives 0.25)")

doe = ROOT / "results" / "doe" / "runs.csv"
if doe.exists():
    ids_d, _, Xd, Yd = wide_table(doe)
    Pd = gp_fit_predict(Xp_n, Yp, add_noise(Xd, cols, 303))
    ed = np.abs(Pd - Yd) / (HI - LO)
    print(f"\nstep 23 devices ({len(Xd)}), S2 model: MAE_norm mean {ed.mean():.4f} | W_JFET {ed[:, 0].mean():.4f}  "
          f"N_pw {ed[:, 1].mean():.4f}  Qit {ed[:, 2].mean():.4f}")
    for i, y, p in list(zip(ids_d, Yd, Pd))[:4]:
        print(f"  {i}: true W {y[0]:.3f} N {y[1]:.3f} Qit {y[2]:.2e} | estimate W {p[0]:.3f} N {p[1]:.3f} Qit {p[2]:.2e}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for j, (t, lab) in enumerate(zip(TARGETS, ("W$_{JFET}$ scale", "N$_{pw}$ scale", "Q$_{it,eff}$ (cm$^{-2}$)"))):
    for name, mk in (("S1 (300 K)", "o"), ("S2 (300 + 423 K)", "s")):
        ax[j].plot(Yt[:, j], pred[name][:, j], mk, ms=3.5, mfc="none",
                   label=f"{name}, MAE$_{{norm}}$ {err[name][:, j].mean():.3f}")
    ax[j].plot([LO[j], HI[j]], [LO[j], HI[j]], "k-", lw=0.8)
    ax[j].set(xlabel=f"true {lab}", ylabel=f"estimated {lab}", title=lab)
    ax[j].legend(fontsize=7)
savefig(fig, IMG / "inverse_parity.png")
