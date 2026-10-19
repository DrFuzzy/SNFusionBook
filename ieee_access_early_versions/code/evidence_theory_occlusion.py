#!/usr/bin/env python3
"""Evidence-theory example for a rain-night autonomous-driving occlusion."""

from __future__ import annotations

from collections import defaultdict


FRAME = ("F", "O", "U")
NAMES = {
    "F": "free lane",
    "O": "occupied by vehicle/obstacle",
    "U": "unknown vulnerable road user",
}
OMEGA = frozenset(FRAME)
EMPTY = frozenset()


def fs(*items: str) -> frozenset[str]:
    return frozenset(items)


def format_set(item: frozenset[str]) -> str:
    if not item:
        return "empty"
    if item == OMEGA:
        return "Omega"
    return "{" + ",".join(sorted(item)) + "}"


def combine_unnormalized(
    left: dict[frozenset[str], float],
    right: dict[frozenset[str], float],
) -> dict[frozenset[str], float]:
    out: defaultdict[frozenset[str], float] = defaultdict(float)
    for a, mass_a in left.items():
        for b, mass_b in right.items():
            out[a & b] += mass_a * mass_b
    return dict(out)


def combine_dempster(
    left: dict[frozenset[str], float],
    right: dict[frozenset[str], float],
) -> tuple[dict[frozenset[str], float], float]:
    unnormalized = combine_unnormalized(left, right)
    conflict = unnormalized.get(EMPTY, 0.0)
    if conflict >= 1.0:
        raise ValueError("Total conflict: Dempster normalization is undefined.")
    scale = 1.0 - conflict
    normalized = {
        focal: mass / scale
        for focal, mass in unnormalized.items()
        if focal and mass > 0.0
    }
    return normalized, conflict


def belief(mass: dict[frozenset[str], float], target: frozenset[str]) -> float:
    return sum(value for focal, value in mass.items() if focal and focal <= target)


def plausibility(mass: dict[frozenset[str], float], target: frozenset[str]) -> float:
    return sum(value for focal, value in mass.items() if focal and focal & target)


def pignistic(mass: dict[frozenset[str], float]) -> dict[str, float]:
    probabilities = {state: 0.0 for state in FRAME}
    for focal, value in mass.items():
        if not focal:
            continue
        share = value / len(focal)
        for state in focal:
            probabilities[state] += share
    return probabilities


def print_mass_table(title: str, mass: dict[frozenset[str], float]) -> None:
    print(title)
    for focal in sorted(mass, key=lambda item: (len(item), sorted(item))):
        print(f"  {format_set(focal):8s}: {mass[focal]:.4f}")
    print()


def main() -> None:
    camera = {
        fs("F"): 0.55,
        fs("O", "U"): 0.20,
        OMEGA: 0.25,
    }
    radar_lidar = {
        fs("O"): 0.45,
        fs("O", "U"): 0.35,
        OMEGA: 0.20,
    }

    unnormalized = combine_unnormalized(camera, radar_lidar)
    dempster, conflict = combine_dempster(camera, radar_lidar)
    betp = pignistic(dempster)

    print("Frame of discernment")
    for state in FRAME:
        print(f"  {state}: {NAMES[state]}")
    print()

    print_mass_table("Camera BBA m_C", camera)
    print_mass_table("Radar-LiDAR BBA m_RL", radar_lidar)
    print_mass_table("Unnormalised Smets combination", unnormalized)
    print(f"Conflict K = {conflict:.4f}\n")
    print_mass_table("Dempster-normalised combination", dempster)

    occupied = fs("O")
    print(f"Bel(O) = {belief(dempster, occupied):.4f}")
    print(f"Pl(O)  = {plausibility(dempster, occupied):.4f}\n")

    print("Pignistic probabilities")
    for state, value in betp.items():
        print(f"  p({state}) = {value:.4f}")


if __name__ == "__main__":
    main()
