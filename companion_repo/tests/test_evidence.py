"""Unit tests for snfusion.evidence (Chapter 3, Evidence Theory)."""

import pytest

from snfusion.evidence import MassFunction, combine_all, conflict_mass, conjunctive_rule, dempster_rule

FRAME = ("F", "O", "U")
OMEGA = frozenset(FRAME)


def _book_bbas():
    camera = MassFunction(FRAME, {"F": 0.55, ("O", "U"): 0.20, OMEGA: 0.25})
    radar_lidar = MassFunction(FRAME, {"O": 0.45, ("O", "U"): 0.35, OMEGA: 0.20})
    return camera, radar_lidar


def test_rain_night_example_numbers():
    camera, rl = _book_bbas()
    smets = camera.combine_conjunctive(rl)
    assert smets.conflict() == pytest.approx(0.44)
    assert conflict_mass(camera, rl) == pytest.approx(0.55 * (0.45 + 0.35))
    fused, K = camera.combine_dempster(rl)
    assert K == pytest.approx(0.44)
    assert fused[frozenset("F")] == pytest.approx(0.1964, abs=5e-5)
    assert fused[frozenset("O")] == pytest.approx(0.3616, abs=5e-5)
    assert fused[frozenset(("O", "U"))] == pytest.approx(0.3527, abs=5e-5)
    assert fused[OMEGA] == pytest.approx(0.0893, abs=5e-5)
    assert sum(fused.values()) == pytest.approx(1.0)
    assert fused.belief("O") == pytest.approx(0.3616, abs=5e-5)
    assert fused.plausibility("O") == pytest.approx(0.8036, abs=5e-5)
    betp = fused.pignistic()
    assert betp["F"] == pytest.approx(0.2262, abs=5e-5)
    assert betp["O"] == pytest.approx(0.5677, abs=5e-5)
    assert betp["U"] == pytest.approx(0.2061, abs=5e-5)
    assert sum(betp.values()) == pytest.approx(1.0)


def test_zadeh_textbook_example():
    # two doctors: m1 = {A: .99, B: .01}, m2 = {C: .99, B: .01}; Dempster says B with certainty
    m1 = MassFunction("ABC", {"A": 0.99, "B": 0.01})
    m2 = MassFunction("ABC", {"C": 0.99, "B": 0.01})
    fused, K = dempster_rule(m1, m2)
    assert K == pytest.approx(0.9999)
    assert fused[frozenset("B")] == pytest.approx(1.0)
    smets = conjunctive_rule(m1, m2)
    assert smets.conflict() == pytest.approx(0.9999)
    assert smets[frozenset("B")] == pytest.approx(1e-4)


def test_total_conflict_raises():
    m1 = MassFunction("AB", {"A": 1.0})
    m2 = MassFunction("AB", {"B": 1.0})
    with pytest.raises(ValueError):
        m1.combine_dempster(m2)


def test_vacuous_bba_is_neutral_and_discounting():
    camera, _ = _book_bbas()
    vac = MassFunction(FRAME, {OMEGA: 1.0})
    fused, K = camera.combine_dempster(vac)
    assert K == 0.0
    for A, v in camera.items():
        assert fused[A] == pytest.approx(v)
    assert camera.discount(0.0) == vac
    assert camera.discount(1.0) == camera
    half = camera.discount(0.5)
    assert sum(half.values()) == pytest.approx(1.0)
    assert half[frozenset("F")] == pytest.approx(0.275)
    assert half[OMEGA] == pytest.approx(0.5 + 0.125)


def test_belief_plausibility_bounds_and_combine_all():
    camera, rl = _book_bbas()
    for m in (camera, rl):
        for A in (("F",), ("O",), ("O", "U"), FRAME):
            assert m.belief(A) <= m.plausibility(A) + 1e-12
        assert m.belief(FRAME) == pytest.approx(1.0)
    d1, _ = camera.combine_dempster(rl)
    d2 = combine_all([camera, rl], rule="dempster")
    assert d1 == d2
    assert combine_all([camera, rl], rule="smets").conflict() == pytest.approx(0.44)


def test_invalid_bba():
    with pytest.raises(ValueError):
        MassFunction("AB", {"A": 0.5})
    with pytest.raises(ValueError):
        MassFunction("AB", {"C": 1.0})
