"""07. Incomplete ionization of N donors (66 meV) and Al acceptors (191 meV) in 4H-SiC.

  python steps/07_ionization.py

In a neutral region with one dopant type: N+ = N / (1 + g n / n1),  n1 = Nc exp(-dE/kT).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.params import SiC4H, ionized_fraction
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "07"
sic = SiC4H()
for T in (300, 423):
    d = sic.derived(T)
    print(f"T = {T} K: n1D = Nc exp(-0.066/kT) = {d['n1D']:.3e} cm^-3, p1A = Nv exp(-0.191/kT) = {d['p1A']:.3e} cm^-3")

rows = [("drift (N)", "donor", 1e16), ("JFET (N)", "donor", 2e16), ("n+ source (N)", "donor", 2e19),
        ("substrate (N)", "donor", 1e19), ("P-well (Al)", "acceptor", 1e17), ("p+ body (Al)", "acceptor", 1e19)]
print(f"\n{'region':<15} {'N (cm-3)':>9} {'ionized 300 K':>14} {'ionized 423 K':>14}")
for name, kind, N in rows:
    f300, f423 = ionized_fraction(N, 300, sic, kind), ionized_fraction(N, 423, sic, kind)
    print(f"{name:<15} {N:9.0e} {100 * f300:13.1f}% {100 * f423:13.1f}%")

N = np.logspace(14, 20, 121)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for T, ls in ((300, "-"), (423, "--")):
    ax.semilogx(N, [100 * ionized_fraction(x, T, sic, "donor") for x in N], ls, color="tab:blue",
                label=f"N donor (66 meV), {T} K")
    ax.semilogx(N, [100 * ionized_fraction(x, T, sic, "acceptor") for x in N], ls, color="tab:red",
                label=f"Al acceptor (191 meV), {T} K")
ax.set(xlabel="dopant density (cm$^{-3}$)", ylabel="ionized fraction (%)", ylim=(0, 102),
       title="4H-SiC, neutral region")
ax.legend(loc="lower left")
savefig(fig, IMG / "ionization.png")
