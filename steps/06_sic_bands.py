"""06. 4H-SiC band gap, effective density of states and intrinsic carrier density vs temperature.

  python steps/06_sic_bands.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.params import SiC4H, KB, Q
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "06"
sic = SiC4H()


def si_ni(T):
    """Silicon for comparison: Eg(T) Varshni (1.170 eV, 4.73e-4, 636 K), Nc, Nv ~ T^1.5."""
    Eg = 1.170 - 4.73e-4 * T**2 / (T + 636.0)
    Nc, Nv = 2.86e19 * (T / 300) ** 1.5, 3.10e19 * (T / 300) ** 1.5
    return math.sqrt(Nc * Nv) * math.exp(-Eg * Q / (2 * KB * T)), Eg


print(f"{'T (K)':>6} {'Vt (V)':>8} {'Eg (eV)':>8} {'Nc (cm-3)':>10} {'Nv (cm-3)':>10} {'ni SiC (cm-3)':>14} "
      f"{'Ec-Ei (eV)':>10} {'ni Si (cm-3)':>13}")
for T in (300, 423, 600):
    d = sic.derived(T)
    print(f"{T:6d} {d['Vt']:8.5f} {d['Eg']:8.4f} {d['Nc']:10.3e} {d['Nv']:10.3e} {d['ni']:14.3e} "
          f"{d['Ec_minus_Ei']:10.4f} {si_ni(T)[0]:13.3e}")

d300 = sic.derived(300)
print("\nminority carriers in equilibrium (p = ni^2 / n):")
for N in (1e16, 2e19):
    print(f"  n-type {N:.0e} cm^-3 at 300 K: p = {d300['ni'] ** 2 / N:.2e} cm^-3")

T = np.linspace(250, 800, 221)
ni_sic = [sic.derived(t)["ni"] for t in T]
ni_si = [si_ni(t)[0] for t in T]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].semilogy(T, ni_sic, label="4H-SiC")
ax[0].semilogy(T, ni_si, "--", label="Si")
for t in (300, 423):
    ax[0].axvline(t, color="gray", lw=0.6, ls=":")
ax[0].axhline(1e16, color="tab:red", lw=0.6)
ax[0].text(260, 2e16, "drift doping 1e16", color="tab:red", fontsize=8)
ax[0].set(xlabel="T (K)", ylabel="n$_i$ (cm$^{-3}$)", title="intrinsic carrier density", ylim=(1e-12, 1e18))
ax[0].legend()
ax[1].plot(T, [sic.derived(t)["Eg"] for t in T], label="4H-SiC")
ax[1].plot(T, [si_ni(t)[1] for t in T], "--", label="Si")
ax[1].set(xlabel="T (K)", ylabel="E$_g$ (eV)", title="band gap")
ax[1].legend()
savefig(fig, IMG / "ni_eg.png")
