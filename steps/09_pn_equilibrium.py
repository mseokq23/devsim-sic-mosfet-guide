"""09. 1D 4H-SiC PN diode in equilibrium: mesh, doping, Poisson-only start, then drift-diffusion.

  python steps/09_pn_equilibrium.py

p+ (Al 1e18, 0-2 um) / n (N 1e16, 2-10 um), 300 K, incomplete ionization on, 128-bit precision.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sicdev.config import load_config, material_from
from sicdev import devices as dv, physics as ph, solver
from sicdev.plot import plt, savefig

IMG = ROOT / "docs" / "img" / "09"
cfg = load_config()
mat = material_from(cfg)
T = 300.0

ph.set_extended_precision(True)
t0 = time.time()
with solver.quiet():
    lay = dv.build_pn_diode("pn", cfg)
    d = dv.setup_physics("pn", lay, mat, T, ii=True)
x = ph.node_array("pn", "sic", "x") * 1e4
print(f"nodes = {len(x)}, smallest spacing = {np.diff(x).min() * 1e3:.1f} nm (at the junction)")

psi_guess = ph.node_array("pn", "sic", "Potential")
solver.solve_dc("pn", dv.POISSON_TOL)
psi_po = ph.node_array("pn", "sic", "Potential")
print(f"step 1, Poisson only   : Potential(left) = {psi_po[0]:+.4f} V, Potential(right) = {psi_po[-1]:+.4f} V")
dv.equilibrate("pn", lay, ii=True, tol=cfg["numerics"]["tol"], holes=True)
psi = ph.node_array("pn", "sic", "Potential")
n = ph.node_array("pn", "sic", "Electrons")
p = ph.node_array("pn", "sic", "Holes")
print(f"step 2, drift-diffusion: Potential(left) = {psi[0]:+.4f} V, Potential(right) = {psi[-1]:+.4f} V")
print(f"max |Potential(DD) - Potential(Poisson only)| = {np.max(np.abs(psi - psi_po)):.2e} V")
print(f"built-in potential = {psi[-1] - psi[0]:.4f} V")
i_p, i_n = np.argmin(abs(x - 0.5)), np.argmin(abs(x - 8.0))
print(f"p-side (x = 0.5 um): p = {p[i_p]:.3e}, n = {n[i_p]:.3e} cm^-3")
print(f"n-side (x = 8.0 um): n = {n[i_n]:.3e}, p = {p[i_n]:.3e} cm^-3")
print(f"time {time.time() - t0:.1f} s, solves {solver.STATS.n_solves}")

# band diagram: Potential is the intrinsic level (Ei = -q*Potential), EF = 0 in equilibrium
Ei = -psi
Ec = Ei + d["Ec_minus_Ei"]
Ev = Ec - d["Eg"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(x, lay["semis"]["sic"][1], label="N$_A$ (Al)")
ax[0].plot(x, lay["semis"]["sic"][0], label="N$_D$ (N)")
ax[0].plot(x, np.full_like(x, 1.3e15), "|", color="k", ms=8, label="mesh nodes")
ax[0].set(yscale="log", ylim=(1e15, 3e18), xlim=(0, 4), xlabel="x (um)", ylabel="cm$^{-3}$", title="doping")
ax[0].legend(loc="center right")
ax[1].plot(x, Ec, label="E$_c$")
ax[1].plot(x, Ei, ":", label="E$_i$")
ax[1].plot(x, Ev, label="E$_v$")
ax[1].axhline(0, color="k", ls="--", lw=0.8, label="E$_F$")
ax[1].set(xlim=(0, 4), xlabel="x (um)", ylabel="energy (eV)", title="band diagram, 300 K")
ax[1].legend(loc="upper right")
ax[2].semilogy(x, n, label="n")
ax[2].semilogy(x, p, label="p")
ax[2].set(xlim=(0, 4), ylim=(1e-40, 1e19), xlabel="x (um)", ylabel="cm$^{-3}$", title="carrier densities")
ax[2].legend()
savefig(fig, IMG / "pn_equilibrium.png")
