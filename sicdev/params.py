"""4H-SiC / SiO2 material parameters and their temperature dependence.

Lengths are in cm (DEVSIM convention) unless the name ends with ``_um``.

Parameter set (the same set as github.com/mseokq23/sic-mosfet-devsim):
  * Eg(T), Nc, Nv        : Levinshtein, Rumyantsev, Shur (2001)
  * electron mobility    : Roschke & Schwierz, IEEE TED 48, 1442 (2001)  (perpendicular to c)
  * hole mobility        : Schaffer et al. (1994), T exponents from Rao (2022)
  * ionization energies  : Ikeda, Matsunami, Tanaka (1980): N (hex site) 66 meV, Al 191 meV
  * permittivity         : 9.7
  * electron affinity    : 3.7 eV (assumed; only Phi_m - chi matters, see gate_offset)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

Q = 1.602176634e-19      # C
KB = 1.380649e-23        # J/K
EPS0 = 8.8541878128e-14  # F/cm


@dataclass
class SiC4H:
    eps_r: float = 9.7
    # band gap, Varshni form anchored at 300 K
    Eg300: float = 3.23
    Eg_alpha: float = 6.5e-4
    Eg_beta: float = 1300.0
    # effective density of states at 300 K, scaled with T^1.5
    Nc300: float = 1.69e19
    Nv300: float = 2.49e19
    chi: float = 3.7
    # incomplete ionization: ionization energy (eV) and degeneracy factor
    dED: float = 0.066
    gD: float = 2.0
    dEA: float = 0.191
    gA: float = 4.0
    # electron mobility, Caughey-Thomas with temperature exponents
    mun_min: float = 40.0
    mun_max: float = 950.0
    mun_Nref: float = 2.0e17
    mun_delta: float = 0.76
    mun_gmin: float = -0.5
    mun_gmax: float = -2.4
    mun_gref: float = 1.0
    # hole mobility
    mup_min: float = 15.9
    mup_max: float = 124.0
    mup_Nref: float = 1.76e19
    mup_delta: float = 0.34
    mup_gmin: float = -0.5
    mup_gmax: float = -2.15
    mup_gref: float = 0.0
    # SRH lifetimes (s)
    taun: float = 1.0e-7
    taup: float = 1.0e-7
    # effective channel (surface) mobility, added with Matthiessen's rule:
    #   1/mu = 1/mu_bulk + exp(-depth/lambda)/mu_surf,  mu_surf = mu_surf300 * (T/300)^gamma
    mu_surf300: float = 20.0
    mu_surf_gamma: float = 1.0
    surf_lambda_um: float = 0.003

    def derived(self, T: float) -> dict:
        """Everything the DEVSIM models need at temperature T (K)."""
        t = T / 300.0
        Vt = KB * T / Q
        Eg = self.Eg300 + self.Eg_alpha * (300.0**2 / (300.0 + self.Eg_beta) - T**2 / (T + self.Eg_beta))
        Nc = self.Nc300 * t**1.5
        Nv = self.Nv300 * t**1.5
        ni = math.sqrt(Nc * Nv) * math.exp(-Eg / (2.0 * Vt))
        Ec_minus_Ei = 0.5 * Eg - 0.5 * Vt * math.log(Nv / Nc)     # eV, intrinsic level below Ec
        return dict(
            T=T, Vt=Vt, Eg=Eg, Nc=Nc, Nv=Nv, ni=ni, Ec_minus_Ei=Ec_minus_Ei,
            n1D=Nc * math.exp(-self.dED / Vt), p1A=Nv * math.exp(-self.dEA / Vt),
            gD=self.gD, gA=self.gA,
            mun_min=self.mun_min * t**self.mun_gmin, mun_max=self.mun_max * t**self.mun_gmax,
            mun_Nref=self.mun_Nref * t**self.mun_gref, mun_delta=self.mun_delta,
            mup_min=self.mup_min * t**self.mup_gmin, mup_max=self.mup_max * t**self.mup_gmax,
            mup_Nref=self.mup_Nref * t**self.mup_gref, mup_delta=self.mup_delta,
            mu_surf=self.mu_surf300 * t**self.mu_surf_gamma,
        )


@dataclass
class SiO2:
    eps_r: float = 3.9


@dataclass
class Gate:
    phi_m: float = 4.1     # n+ poly-Si gate work function (eV)


def gate_offset(mat: SiC4H, gate: Gate, T: float) -> float:
    """Phi_m - Phi_i (V).  Gate boundary condition: Potential = V_G - gate_offset.

    The potential is measured from the intrinsic level (n = p = n_i at Potential = 0),
    so the semiconductor work function seen by the gate is chi + (Ec - Ei).
    """
    d = mat.derived(T)
    return gate.phi_m - (mat.chi + d["Ec_minus_Ei"])


def ionized_fraction(N, T, mat: SiC4H | None = None, kind="donor"):
    """Ionized fraction of a dopant in a neutral region (charge balance with one carrier type).

    donor   : n = N+ = N / (1 + gD n / n1D)  ->  gD/n1D * n^2 + n - N = 0
    acceptor: same with gA, p1A
    """
    mat = mat or SiC4H()
    d = mat.derived(T)
    a = (d["gD"] / d["n1D"]) if kind == "donor" else (d["gA"] / d["p1A"])
    n = (-1.0 + math.sqrt(1.0 + 4.0 * a * N)) / (2.0 * a)
    return n / N


def parameter_table(mat: SiC4H | None = None) -> list[tuple[str, float]]:
    return list(asdict(mat or SiC4H()).items())
