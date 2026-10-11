"""Evidence theory (Dempster--Shafer / Transferable Belief Model).

Chapter 3, Section "Evidence Theory for Conflicting Sensor Information":

* basic belief assignments over a frame of discernment, eq. (bba);
* belief ``Bel`` and plausibility ``Pl``;
* Dempster's normalised rule, eq. (dempster), with the conflict ``K``;
* Smets' unnormalised conjunctive rule (TBM), eq. (smets), ``m(empty) = K``;
* reliability discounting ``m^alpha``;
* the pignistic transform, eq. (pignistic-transform).

Focal elements are ``frozenset`` objects; :class:`MassFunction` is a thin
mapping ``frozenset -> mass`` that keeps the frame alongside the masses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, Mapping, Tuple, Union

Focal = frozenset
MassDict = Dict[Focal, float]
SetLike = Union[Hashable, Iterable[Hashable], Focal]


def _as_set(item: SetLike) -> Focal:
    if isinstance(item, frozenset):
        return item
    if isinstance(item, (str, bytes)) or not isinstance(item, Iterable):
        return frozenset([item])
    return frozenset(item)


def format_set(item: Focal, frame: Focal) -> str:
    """Pretty print a focal element (``Omega`` for the whole frame)."""
    if not item:
        return "empty"
    if item == frame:
        return "Omega"
    return "{" + ",".join(str(e) for e in sorted(item, key=str)) + "}"


class MassFunction(dict):
    """A basic belief assignment ``m : 2^Omega -> [0, 1]``.

    Parameters
    ----------
    frame : the hypotheses ``Omega``.
    masses : mapping ``focal element -> mass``; keys may be single hypotheses,
             iterables of hypotheses or frozensets.  Masses must sum to one.

    Example (camera BBA of the rain-night example)
    ------------------------------------------------
    >>> m = MassFunction(("F", "O", "U"), {"F": 0.55, ("O", "U"): 0.20, ("F", "O", "U"): 0.25})
    >>> round(m.belief(("O", "U")), 2), round(m.plausibility("O"), 2)
    (0.2, 0.45)
    """

    def __init__(self, frame: Iterable[Hashable], masses: Mapping[SetLike, float],
                 check: bool = True, tol: float = 1e-9) -> None:
        super().__init__()
        self.frame: Focal = frozenset(frame)
        for key, value in masses.items():
            A = _as_set(key)
            if not A <= self.frame:
                raise ValueError(f"focal element {set(A)} is not a subset of the frame")
            if value != 0.0:
                self[A] = self.get(A, 0.0) + float(value)
        if check:
            total = sum(self.values())
            if abs(total - 1.0) > tol:
                raise ValueError(f"masses must sum to one (got {total})")
            if any(v < -tol for v in self.values()):
                raise ValueError("masses must be non-negative")

    # ---- derived quantities -----------------------------------------------------
    def focal_elements(self):
        return [A for A in self if self[A] > 0.0]

    def belief(self, A: SetLike) -> float:
        """``Bel(A) = sum_{B subseteq A, B != empty} m(B)``."""
        A = _as_set(A)
        return sum(v for B, v in self.items() if B and B <= A)

    def plausibility(self, A: SetLike) -> float:
        """``Pl(A) = sum_{B cap A != empty} m(B)``."""
        A = _as_set(A)
        return sum(v for B, v in self.items() if B & A)

    def conflict(self) -> float:
        """Mass on the empty set (non-zero only after a TBM combination)."""
        return self.get(frozenset(), 0.0)

    def pignistic(self) -> Dict[Hashable, float]:
        """Pignistic probability ``BetP(w) = sum_{A ∋ w} m(A)/|A|``, eq. (pignistic-transform).

        Under the open-world assumption (``m(empty) > 0``) the result is rescaled
        by ``1/(1 - m(empty))`` as stated in the book.
        """
        betp = {w: 0.0 for w in self.frame}
        scale = 1.0 - self.conflict()
        for A, v in self.items():
            if not A:
                continue
            share = v / len(A) / scale
            for w in A:
                betp[w] += share
        return betp

    # ---- combination rules -----------------------------------------------------
    def combine_conjunctive(self, other: "MassFunction") -> "MassFunction":
        """Smets' unnormalised conjunctive rule, eq. (smets): ``m(empty)`` is the conflict ``K``."""
        if other.frame != self.frame:
            raise ValueError("mass functions must share the same frame")
        out: Dict[Focal, float] = defaultdict(float)
        for A, ma in self.items():
            for B, mb in other.items():
                out[A & B] += ma * mb
        return MassFunction(self.frame, dict(out), check=False)

    def combine_dempster(self, other: "MassFunction") -> Tuple["MassFunction", float]:
        """Dempster's rule, eq. (dempster). Returns ``(m_1 (+) m_2, K)``."""
        unnormalised = self.combine_conjunctive(other)
        K = unnormalised.conflict()
        if K >= 1.0:
            raise ValueError("total conflict: Dempster's normalisation is undefined")
        scale = 1.0 - K
        fused = {A: v / scale for A, v in unnormalised.items() if A and v > 0.0}
        return MassFunction(self.frame, fused), K

    def discount(self, alpha: float) -> "MassFunction":
        """Reliability discounting: ``m^a(A) = a m(A)`` for ``A != Omega`` and
        ``m^a(Omega) = 1 - a + a m(Omega)``."""
        out = {A: alpha * v for A, v in self.items() if A != self.frame}
        out[self.frame] = 1.0 - alpha + alpha * self.get(self.frame, 0.0)
        return MassFunction(self.frame, out)

    def table(self) -> str:
        """Multi-line string with one focal element per line (sorted by size)."""
        lines = []
        for A in sorted(self, key=lambda s: (len(s), sorted(map(str, s)))):
            lines.append(f"  {format_set(A, self.frame):8s}: {self[A]:.4f}")
        return "\n".join(lines)


# ---- functional interface (convenience wrappers) ------------------------------------
def dempster_rule(m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
    """Functional alias of :meth:`MassFunction.combine_dempster`."""
    return m1.combine_dempster(m2)


def conjunctive_rule(m1: MassFunction, m2: MassFunction) -> MassFunction:
    """Functional alias of :meth:`MassFunction.combine_conjunctive` (Smets/TBM)."""
    return m1.combine_conjunctive(m2)


def conflict_mass(m1: MassFunction, m2: MassFunction) -> float:
    """``K = sum_{A cap B = empty} m1(A) m2(B)``."""
    return m1.combine_conjunctive(m2).conflict()


def combine_all(masses: Iterable[MassFunction], rule: str = "dempster") -> MassFunction:
    """Fold a sequence of BBAs with Dempster's (``"dempster"``) or Smets' (``"smets"``) rule."""
    it = iter(masses)
    acc = next(it)
    for m in it:
        if rule == "dempster":
            acc, _ = acc.combine_dempster(m)
        elif rule == "smets":
            acc = acc.combine_conjunctive(m)
        else:
            raise ValueError("rule must be 'dempster' or 'smets'")
    return acc
