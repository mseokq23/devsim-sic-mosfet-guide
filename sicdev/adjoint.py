"""Parameter sensitivities of a terminal current with the adjoint method.

At a converged bias point the Newton residual is F(u, theta) = 0. DEVSIM's get_matrix_and_rhs()
gives the Jacobian J = dF/du and the Newton right-hand side R = -F at the present solution.
Differentiating F(u(theta), theta) = 0:

    J du/dtheta = dR/dtheta                                  (one solve per parameter)
    dI/dtheta = e^T du/dtheta = lam^T dR/dtheta,  J^T lam = e   (one solve for all parameters)

I = u[k] must be a solution unknown: a contact tied to a circuit voltage source makes its current
the circuit unknown "<source>.I". dR/dtheta comes from two residual assemblies at theta +- h around
the converged solution (no nonlinear re-solve).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import devsim as ds


def assemble():
    """(J, R) at the present solution, J as scipy CSC."""
    m = ds.get_matrix_and_rhs(format="csc")["static"]
    n = len(m["rhs"])
    J = sp.csc_matrix((np.asarray(m["av"], float), np.asarray(m["ai"]), np.asarray(m["ap"])), shape=(n, n))
    return J, np.asarray(m["rhs"], float)


class ScaledLU:
    """Sparse LU of the row/column-scaled Jacobian, with two steps of iterative refinement.
    The unknowns mix potentials (~1 V) and carrier densities (up to ~1e20 cm^-3); without scaling
    a plain LU loses several digits."""

    def __init__(self, J, refine=2):
        self.J, self.JT = J.tocsc(), J.T.tocsc()
        A = J.tocsr()
        r = np.asarray(abs(A).max(axis=1).todense()).ravel()
        r[r == 0] = 1.0
        self.dr = 1.0 / r
        A1 = sp.diags(self.dr) @ A
        c = np.asarray(abs(A1).max(axis=0).todense()).ravel()
        c[c == 0] = 1.0
        self.dc = 1.0 / c
        self.lu = spla.splu((A1 @ sp.diags(self.dc)).tocsc())
        self.refine = refine

    def solve(self, b, transpose=False):
        b = np.asarray(b, float)
        if not transpose:
            x = self.dc * self.lu.solve(self.dr * b)
            for _ in range(self.refine):
                x = x + self.dc * self.lu.solve(self.dr * (b - self.J @ x))
        else:
            x = self.dr * self.lu.solve(self.dc * b, trans="T")
            for _ in range(self.refine):
                x = x + self.dr * self.lu.solve(self.dc * (b - self.JT @ x), trans="T")
        return x


def residual_derivative(set_value, value, h):
    """dR/dtheta by a central difference of two residual assemblies; set_value(v) changes theta only."""
    try:
        set_value(value + h)
        rp = assemble()[1]
        set_value(value - h)
        rm = assemble()[1]
    finally:
        set_value(value)
    return (rp - rm) / (2.0 * h)


def adjoint_gradient(output_unknown: str, params: dict):
    """d(output)/d(theta) for every theta in params = {name: (set_value, value, h)}.

    output_unknown: circuit unknown such as "V1.I". Returns ({name: gradient}, lam)."""
    J, _ = assemble()
    lu = ScaledLU(J)
    e = np.zeros(J.shape[0])
    e[ds.get_circuit_equation_number(node=output_unknown)] = 1.0
    lam = lu.solve(e, transpose=True)
    return {name: float(lam @ residual_derivative(*p)) for name, p in params.items()}, lam
