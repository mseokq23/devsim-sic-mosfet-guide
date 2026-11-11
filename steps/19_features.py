"""19. The fixed run protocol (sicdev.mosfet.run_mosfet) and feature extraction, 300 K.

  python steps/19_features.py      -> results/base_T300.json
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config
from sicdev.mosfet import run_mosfet
from sicdev.utils import save_json
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "19"
cfg = load_config()
ex = cfg["extract"]
rec = run_mosfet(cfg, 300.0)
save_json(rec, ROOT / "results" / "base_T300.json")
f = rec["features"]
print(f"converged {rec['converged']}, {rec['n_solves']} solves ({rec['n_fail']} failed), {rec['runtime_s']:.1f} s, "
      f"{rec['n_nodes']} nodes")
print("bias-refined gate voltages (secant search):",
      ", ".join(f"ID = {k} A/cm at VGS = {v:.4f} V" for k, v in rec["vg_at_current"].items()))
rows = [("Vth (ID = 1e-4 A/cm, refined)", f["vth_V"], "V"), ("Vth (interpolated from the curve)", f["vth_interp_V"], "V"),
        ("SS (1e-10 .. 1e-6 A/cm, refined)", f["ss_mV_dec"], "mV/dec"),
        ("SS (interpolated)", f["ss_interp_mV_dec"], "mV/dec"),
        ("gm,max", f["gm_max_S_per_cm"], "S/cm"), ("VGS at gm,max", f["vg_at_gm_max_V"], "V"),
        ("Ion (VGS = 18 V, VDS = 0.1 V)", f["ion_A_per_cm"], "A/cm"), ("Ron,sp", f["ron_mohm_cm2"], "mOhm cm^2"),
        ("ID (VGS = 18 V, VDS = 2 V)", f["id_vds_req_A_per_cm"], "A/cm"),
        ("numerical floor max|ID + IS|", f["kcl_floor_A_per_cm"], "A/cm")]
for name, v, unit in rows:
    print(f"  {name:<36} {v:12.5g} {unit}")
print("  log10 ID samples:", ", ".join(f"{k[8:]} V: {f[k]:.3f}" for k in f if k.startswith("logid_vg")))

tr, out = rec["curves"]["transfer"], rec["curves"]["output"]
vg, idd = np.array(tr["vgs"]), np.array(tr["id"])
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
ax[0].semilogy(vg, np.abs(idd), ".-", ms=3)
for t in ex["ss_window_A_per_cm"]:
    ax[0].axhline(t, color="gray", ls=":", lw=0.8)
ax[0].axhline(ex["icc_A_per_cm"], color="tab:red", lw=0.8)
ax[0].plot([f["vth_V"]], [ex["icc_A_per_cm"]], "o", color="tab:red")
ax[0].text(f["vth_V"] + 0.4, ex["icc_A_per_cm"] * 0.15, f"Vth = {f['vth_V']:.3f} V", color="tab:red", fontsize=8)
ax[0].text(9, 3e-9, f"SS window, SS = {f['ss_mV_dec']:.1f} mV/dec", color="gray", fontsize=8)
for k in [k for k in f if k.startswith("logid_vg")]:
    ax[0].plot(float(k[8:]), 10 ** f[k], "x", color="tab:green", ms=5)
ax[0].set(xlim=(2, 20.5), ylim=(1e-12, 1e-1), xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (A/cm)",
          title="(a) Vth, SS, log I$_D$ samples (x)")
gm = np.gradient(idd, vg)
ax[1].plot(vg, idd * 1e3, ".-", ms=3, label="I$_D$")
a2 = ax[1].twinx()
a2.plot(vg, gm * 1e3, color="tab:orange", lw=1, label="g$_m$")
a2.plot([f["vg_at_gm_max_V"]], [f["gm_max_S_per_cm"] * 1e3], "o", color="tab:orange")
a2.set_ylabel("g$_m$ (mS/cm)", color="tab:orange")
a2.grid(False)
ax[1].plot([18], [f["ion_A_per_cm"] * 1e3], "s", color="tab:red")
ax[1].text(11.5, f["ion_A_per_cm"] * 1e3 * 0.55, f"Ion = {f['ion_A_per_cm'] * 1e3:.2f} mA/cm", color="tab:red", fontsize=8)
ax[1].set(xlabel="V$_{GS}$ (V)", ylabel="I$_D$ (mA/cm)", title="(b) g$_m$ = dI$_D$/dV$_{GS}$, Ion")
vd, io = np.array(out["vds"]), np.array(out["id"])
ax[2].plot(vd, io, "o-", ms=3)
m = vd <= ex["ron_vds_max"] + 1e-9
gfit = np.sum(vd[m] * io[m]) / np.sum(vd[m] ** 2)
ax[2].plot([0, 1.0], [0, gfit], "--", color="tab:red", label=f"fit V$_{{DS}}$ <= 0.5 V: Ron,sp = {f['ron_mohm_cm2']:.3f} m$\\Omega$cm$^2$")
ax[2].plot([2.0], [f["id_vds_req_A_per_cm"]], "s", color="tab:green", label=f"I$_D$(2 V) = {f['id_vds_req_A_per_cm']:.3f} A/cm")
ax[2].set(xlabel="V$_{DS}$ (V)", ylabel="I$_D$ (A/cm)", title="(c) Ron,sp and I$_D$(V$_{DS}$ = 2 V)")
ax[2].legend(fontsize=7)
savefig(fig, IMG / "features.png")
