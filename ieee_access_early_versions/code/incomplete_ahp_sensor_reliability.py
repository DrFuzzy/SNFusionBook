#!/usr/bin/env python3
"""Incomplete-AHP example for autonomous-driving sensor-stream reliability."""

from __future__ import annotations

import math

import numpy as np


ALTERNATIVES = [
    ("C", "camera semantics"),
    ("L", "LiDAR geometry"),
    ("R", "radar range-rate"),
    ("M", "map/IMU lane prior"),
]

# Each tuple means "left is ratio times as preferred as right".
COMPARISONS = [
    ("L", "C", 2.0),
    ("R", "C", 5.0),
    ("M", "C", 3.0),
    ("R", "M", 1.5),
]


def solve_incomplete_lls() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    index = {code: i for i, (code, _) in enumerate(ALTERNATIVES)}
    n = len(ALTERNATIVES)
    design = []
    target = []

    for left, right, ratio in COMPARISONS:
        row = np.zeros(n)
        row[index[left]] = 1.0
        row[index[right]] = -1.0
        design.append(row)
        target.append(math.log(ratio))

    design_matrix = np.vstack(design)
    target_vector = np.array(target)

    # Minimise ||B y - b||^2 subject to sum_i y_i = 0.
    normal_matrix = design_matrix.T @ design_matrix
    gauge = np.ones((n, 1))
    kkt = np.block(
        [
            [normal_matrix, gauge],
            [gauge.T, np.zeros((1, 1))],
        ]
    )
    rhs = np.concatenate([design_matrix.T @ target_vector, np.array([0.0])])
    solution = np.linalg.solve(kkt, rhs)
    log_priorities = solution[:n]
    priorities = np.exp(log_priorities)
    priorities /= priorities.sum()
    return log_priorities, priorities, index


def fitted_ratio(log_priorities: np.ndarray, index: dict[str, int], left: str, right: str) -> float:
    return float(np.exp(log_priorities[index[left]] - log_priorities[index[right]]))


def main() -> None:
    log_priorities, priorities, index = solve_incomplete_lls()

    print("Incomplete AHP priorities for a rain-night autonomous-driving scene")
    for (code, label), priority in zip(ALTERNATIVES, priorities):
        print(f"  {code:1s} ({label:20s}) = {priority:.4f}")
    print()

    print("Observed comparisons and fitted ratios")
    for left, right, ratio in COMPARISONS:
        fitted = fitted_ratio(log_priorities, index, left, right)
        print(f"  A_{{{left},{right}}}: observed {ratio:.3f}, fitted {fitted:.3f}")
    print()

    print("Ratios inferred from the incomplete graph")
    for left, right in [("R", "L"), ("M", "L")]:
        fitted = fitted_ratio(log_priorities, index, left, right)
        print(f"  A_{{{left},{right}}} ~= {fitted:.3f}")


if __name__ == "__main__":
    main()
