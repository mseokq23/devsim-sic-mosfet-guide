"""04. Why Scharfetter-Gummel: 1D electron continuity in a constant field, central difference vs SG.

  python steps/04_sg_vs_central.py

Electron current (normalised, q*mu = 1):  J = Vt dn/dx - n dpsi/dx,  dJ/dx = 0,  psi linear.
Boundary values n(0) = 1, n(L) = 0. Exact: n(x) = (exp(a x) - exp(a L)) / (1 - exp(a L)), a = psi'/Vt.
Edge flux between nodes i, i+1 (d = (psi_i+1 - psi_i)/Vt):
  central : J = Vt/h * [(n_i+1 - n_i) - d (n_i + n_i+1)/2]
  SG      : J = Vt/h * [n_i+1 B(d) - n_i B(-d)],  B(x) = x / (exp(x) - 1)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "04"
VT = 0.025852


def bern(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = x[m] / np.expm1(x[m])
    return out


def solve(n_cells, drop_vt, scheme):
    """Nodes 0..N, potential falls linearly by drop_vt*Vt from left to right (electrons pushed right)."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    h = x[1] - x[0]
    psi = drop_vt * VT * x                         # psi rises to the right: electrons drift right
    d = np.diff(psi) / VT                          # per edge
    if scheme == "central":
        a_next = (1.0 - d / 2.0) * VT / h          # coefficient of n_{i+1} in J_{i+1/2}
        a_this = -(1.0 + d / 2.0) * VT / h         # coefficient of n_i
    else:
        a_next = bern(d) * VT / h
        a_this = -bern(-d) * VT / h
    N = n_cells + 1
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0, 0] = 1.0
    b[0] = 1.0
    A[-1, -1] = 1.0
    b[-1] = 0.0
    for i in range(1, N - 1):               # J_{i+1/2} - J_{i-1/2} = 0
        A[i, i + 1] += a_next[i]
        A[i, i] += a_this[i]
        A[i, i] -= a_next[i - 1]
        A[i, i - 1] -= a_this[i - 1]
    return x, np.linalg.solve(A, b)


def exact(x, drop_vt):
    a = drop_vt          # d(psi)/dx / Vt in units of 1/L
    return (np.exp(a * x) - np.exp(a)) / (1.0 - np.exp(a))


drop = 20.0   # total potential change = 20 Vt (about 0.52 V)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
xf = np.linspace(0, 1, 801)
print(f"total potential change = {drop:.0f} Vt = {drop * VT:.3f} V")
print(f"{'cells':>5} {'dpsi/Vt per edge':>17} {'max |err| central':>18} {'max |err| SG':>13} {'max n central':>14}")
for a, n_cells in zip(ax, (40, 8, 4)):
    xc, nc = solve(n_cells, drop, "central")
    xs, ns = solve(n_cells, drop, "sg")
    ne = exact(xs, drop)
    print(f"{n_cells:5d} {drop / n_cells:17.1f} {np.max(np.abs(nc - ne)):18.3e} {np.max(np.abs(ns - ne)):13.3e} "
          f"{nc.max():14.3f}")
    a.plot(xf, exact(xf, drop), "k-", lw=1, label="exact")
    a.plot(xc, nc, "o--", color="tab:red", label="central difference")
    a.plot(xs, ns, "s:", color="tab:blue", mfc="none", label="Scharfetter-Gummel")
    a.set(title=f"{n_cells} cells, {drop / n_cells:.1f} Vt per edge", xlabel="x / L")
    a.axhline(0, color="gray", lw=0.6)
ax[0].set_ylabel("n / n(0)")
ax[0].legend()
savefig(fig, IMG / "sg_vs_central.png")

# Bernoulli function
x = np.linspace(-6, 6, 241)
fig, a = plt.subplots(figsize=(4.6, 3.4))
a.plot(x, bern(x), label="B(x) = x / (exp(x) - 1)")
a.plot(x, bern(-x), "--", label="B(-x) = B(x) + x")
a.set(xlabel="x = (psi_i+1 - psi_i) / Vt", ylabel="B", ylim=(0, 6.5))
a.legend()
savefig(fig, IMG / "bernoulli.png")
