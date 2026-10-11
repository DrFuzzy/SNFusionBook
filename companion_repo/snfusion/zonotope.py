"""Zonotopes for set-membership (bounded-error) estimation.

Chapter 3, Section "Set-Membership and Zonotopic Estimation".  A zonotope is
written in centre--generator form ``Z = p (+) H B_inf^m = {p + H xi : xi in [-1,1]^m}``.
The class implements

* affine maps and Minkowski sums (exact in this representation),
* the single-strip correction ``p_hat = p + lambda (y - c^T p)``,
  ``H_hat = [(I - lambda c^T) H,  phi lambda]`` with either a user gain or the
  Frobenius-optimal gain of eq. (zono-optimal-gain),
  ``lambda* = H H^T c / (c^T H H^T c + phi^2)``,
* order reduction by interval-hull compression of the smallest generators
  (``H = [H_keep, D]``, ``D = diag(sum_j |(H_red)_ij|)``),
* the interval hull (axis-aligned bounds).

The operation order of :meth:`Zonotope.strip_intersection` and
:meth:`Zonotope.reduce` is that of the authors' ``zono_radar_cutin.py`` so the
companion example reproduces the book's CSV exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


class Zonotope:
    """A zonotope ``p (+) H B_inf^m`` in ``R^n``.

    Parameters
    ----------
    center : ``(n,)`` centre ``p``.
    generators : ``(n, m)`` generator matrix ``H`` (``m`` may be 0).
    """

    __array_ufunc__ = None   # let numpy defer ``A @ Z`` to :meth:`__rmatmul__`

    def __init__(self, center: Array, generators: Array) -> None:
        self.center = np.asarray(center, dtype=float).reshape(-1)
        G = np.asarray(generators, dtype=float)
        if G.ndim == 1:
            G = G.reshape(self.center.size, -1)
        if G.shape[0] != self.center.size:
            raise ValueError("generator matrix must have one row per state dimension")
        self.generators = G

    # ---- constructors -----------------------------------------------------------
    @classmethod
    def from_box(cls, center: Array, half_widths: Array) -> "Zonotope":
        """Axis-aligned box ``[center - hw, center + hw]`` as a zonotope."""
        return cls(center, np.diag(np.asarray(half_widths, dtype=float)))

    # ---- basic properties ------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.center.size

    @property
    def order(self) -> int:
        """Number of generators ``m`` (the "generator count" of the book)."""
        return self.generators.shape[1]

    def __repr__(self) -> str:
        return f"Zonotope(dim={self.dim}, order={self.order}, center={self.center})"

    # ---- exact operations ------------------------------------------------------
    def linear_map(self, A: Array, b: Optional[Array] = None) -> "Zonotope":
        """Affine image ``A Z + b = (A p + b) (+) A H B_inf``."""
        A = np.asarray(A, dtype=float)
        c = A @ self.center
        if b is not None:
            c = c + np.asarray(b, dtype=float)
        return Zonotope(c, A @ self.generators)

    def minkowski_sum(self, other: "Zonotope") -> "Zonotope":
        """``Z_a (+) Z_b = (p_a + p_b) (+) [H_a H_b] B_inf``."""
        return Zonotope(self.center + other.center,
                        np.hstack([self.generators, other.generators]))

    def __add__(self, other: "Zonotope") -> "Zonotope":
        return self.minkowski_sum(other)

    def __rmatmul__(self, A: Array) -> "Zonotope":
        return self.linear_map(A)

    def predict(self, F: Array, process_generators: Array) -> "Zonotope":
        """Exact linear prediction ``F Z (+) (0 (+) G_q B_inf)`` (Example ex:zono-radar-cutin)."""
        return Zonotope(F @ self.center, np.hstack([F @ self.generators, process_generators]))

    # ---- bounds ---------------------------------------------------------------
    def half_widths(self) -> Array:
        """Half-widths of the interval hull, ``sum_j |H_ij|``."""
        return np.abs(self.generators).sum(axis=1)

    def interval_hull(self) -> Tuple[Array, Array]:
        """Axis-aligned bounds ``(lo, hi)`` enclosing the zonotope."""
        hw = self.half_widths()
        return self.center - hw, self.center + hw

    def contains(self, x: Array, tol: float = 1e-9) -> bool:
        """Containment test on the interval hull (sufficient for the book's checks).

        For an exact membership test solve the LP ``|xi|_inf <= 1, H xi = x - p``;
        :meth:`contains_exact` does so with scipy.
        """
        return bool(np.all(np.abs(np.asarray(x, dtype=float) - self.center) <= self.half_widths() + tol))

    def contains_exact(self, x: Array, tol: float = 1e-9) -> bool:
        """Exact membership via a feasibility LP (requires scipy)."""
        from scipy.optimize import linprog

        H = self.generators
        rhs = np.asarray(x, dtype=float) - self.center
        m = H.shape[1]
        if m == 0:
            return bool(np.allclose(rhs, 0.0, atol=tol))
        res = linprog(np.zeros(m), A_eq=H, b_eq=rhs, bounds=[(-1.0, 1.0)] * m, method="highs")
        return bool(res.status == 0)

    def radius_frobenius(self) -> float:
        """Frobenius radius ``||H||_F``, the size measure minimised by the optimal gain."""
        return float(np.linalg.norm(self.generators))

    # ---- measurement correction ------------------------------------------------
    def strip_intersection(self, c: Array, y: float, phi: float,
                           gain: Optional[Array] = None) -> "Zonotope":
        """Outer-approximate ``Z ∩ {x : |c^T x - y| <= phi}`` by a zonotope.

        With ``gain=None`` the Frobenius-optimal gain of eq. (zono-optimal-gain)
        is used; otherwise the supplied ``lambda`` (e.g. ``gain = c`` in Example
        ex:zonotope-strip).  Returns ``p_hat (+) [(I - lambda c^T) H, phi lambda] B_inf``.
        """
        c = np.asarray(c, dtype=float).reshape(-1)
        p, H = self.center, self.generators
        HHt_c = H @ (H.T @ c)
        if gain is None:
            lam = HHt_c / (c @ HHt_c + phi ** 2)
        else:
            lam = np.asarray(gain, dtype=float).reshape(-1)
        p_new = p + lam * (y - c @ p)
        H_new = np.hstack([H - np.outer(lam, H.T @ c), (phi * lam)[:, None]])
        return Zonotope(p_new, H_new)

    def intersect_strips(self, strips, gain=None) -> "Zonotope":
        """Sequentially apply :meth:`strip_intersection` to ``[(c, y, phi), ...]``."""
        Z = self
        for c, y, phi in strips:
            Z = Z.strip_intersection(c, y, phi, gain)
        return Z

    # ---- order reduction ------------------------------------------------------
    def reduce(self, max_generators: int) -> "Zonotope":
        """Interval-hull order reduction preserving containment.

        Keeps the ``max_generators - n`` largest-norm columns and replaces the
        remaining ones by the diagonal interval generator ``D`` of the book.
        """
        H = self.generators
        n, m = H.shape
        if m <= max_generators:
            return self
        if max_generators < n:
            raise ValueError("max_generators must be at least the state dimension")
        order = np.argsort(-np.linalg.norm(H, axis=0))
        keep, red = H[:, order[:max_generators - n]], H[:, order[max_generators - n:]]
        return Zonotope(self.center, np.hstack([keep, np.diag(np.abs(red).sum(axis=1))]))

    # ---- geometry helpers ------------------------------------------------------
    def vertices_2d(self) -> Array:
        """Vertices of a planar zonotope in counter-clockwise order (for plotting)."""
        if self.dim != 2:
            raise ValueError("vertices_2d is only defined for planar zonotopes")
        G = self.generators.copy()
        G[:, G[1] < 0] *= -1                      # orient generators to the upper half-plane
        G = G[:, np.argsort(np.arctan2(G[1], G[0]))]
        start = self.center - G.sum(axis=1)
        pts = [start]
        for g in G.T:
            pts.append(pts[-1] + 2 * g)
        for g in G.T:
            pts.append(pts[-1] - 2 * g)
        return np.array(pts[:-1])
