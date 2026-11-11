"""08. Electron mobility: Caughey-Thomas bulk model and the effective surface (channel) term.

  python steps/08_mobility.py

bulk   : mu = mu_min + (mu_max - mu_min) / (1 + (N/N_ref)^delta), each coefficient ~ (T/300)^gamma_i
surface: 1/mu = 1/mu_bulk + exp(-depth/lambda) / mu_surf,  mu_surf = 20 (T/300)^1 cm^2/Vs, lambda = 3 nm
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "08"
mat = material_from(load_config())


def mu_bulk(N, d):
    return d["mun_min"] + (d["mun_max"] - d["mun_min"]) / (1 + (N / d["mun_Nref"]) ** d["mun_delta"])


print(f"{'N (cm-3)':>9} {'mu_n 300 K':>11} {'mu_n 423 K':>11} {'ratio':>6}")
for N in (1e16, 2e16, 1.1e17, 2e19):
    m3, m4 = mu_bulk(N, mat.derived(300)), mu_bulk(N, mat.derived(423))
    print(f"{N:9.1e} {m3:11.1f} {m4:11.1f} {m4 / m3:6.2f}")
for T in (300, 423):
    d = mat.derived(T)
    mb = mu_bulk(1.1e17, d)
    print(f"T = {T} K: mu_surf = {d['mu_surf']:.1f} cm2/Vs, mu at the surface (P-well 1e17 + 1e16) = "
          f"{1 / (1 / mb + 1 / d['mu_surf']):.1f}, at 3 nm = {1 / (1 / mb + np.exp(-1) / d['mu_surf']):.1f}, "
          f"at 10 nm = {1 / (1 / mb + np.exp(-10 / 3) / d['mu_surf']):.1f}")

N = np.logspace(14, 20, 121)
depth = np.linspace(0, 20, 201)  # nm
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for T, ls in ((300, "-"), (423, "--")):
    d = mat.derived(T)
    ax[0].semilogx(N, mu_bulk(N, d), ls, label=f"{T} K")
    mb = mu_bulk(1.1e17, d)
    ax[1].plot(depth, 1 / (1 / mb + np.exp(-depth / 3.0) / d["mu_surf"]), ls, label=f"{T} K")
ax[0].set(xlabel="N$_D$ + N$_A$ (cm$^{-3}$)", ylabel="$\\mu_n$ (cm$^2$/Vs)", title="bulk electron mobility")
ax[1].set(xlabel="depth from the SiO$_2$/SiC interface (nm)", ylabel="$\\mu_n$ (cm$^2$/Vs)",
          title="channel region (P-well), with surface term", yscale="log")
for a in ax:
    a.legend()
savefig(fig, IMG / "mobility.png")
