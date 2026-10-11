# Chapter 3 — Classical Sensor Fusion Techniques: problems

End-of-chapter problems of Chapter 3 of *Towards a Unified View of Sensor Fusion: Theory, Methods, and Autonomous Driving Applications* (Deliparaschos, Oliva, Charalambous). The printed solutions are collected at the end of the book; every number quoted there is reproduced by the scripts in `solutions/`:

| file | content |
|---|---|
| `solutions/ch03_hand_computations.py` / `.ipynb` | numerical verification of Problems 3.1–3.9 (one cell per problem) |
| `solutions/ch03_prob10_ekf_process_noise.py` / `.ipynb` | Problem 3.10: radar cut-in EKF with a sweep of the process-noise intensity; prints the table of the printed solution and writes `ch03_prob10_nis_vs_sigma_a.png` |

The scripts are self-contained (numpy, scipy, matplotlib only), deterministic, and use `# %%` cell markers; the notebooks are generated with `python3 tools/py2nb.py <script.py>` from the repository root.

Equation numbers below refer to the book.

## Problem 3.1 — Least squares versus weighted least squares

A radar and a camera both report the longitudinal distance *x* to a lead vehicle: *y₁* = 41.6 m with σ₁ = 0.30 m and *y₂* = 43.4 m with σ₂ = 1.50 m, the two errors being zero-mean and independent.

1. Write the stacked linear model **y** = **G**x + **r** with **G** = [1 1]ᵀ, **R** = diag(σ₁², σ₂²), and compute the ordinary least-squares estimate and its variance (**G**ᵀ**G**)⁻¹**G**ᵀ**R****G**(**G**ᵀ**G**)⁻¹.
2. Compute the information Λ = **G**ᵀ**R**⁻¹**G**, the WLS estimate and its variance Λ⁻¹.
3. Show that in general x̂_WLS = (σ₁⁻² y₁ + σ₂⁻² y₂)/(σ₁⁻² + σ₂⁻²), the information-weighted average, and that its variance never exceeds that of the better sensor alone.
4. By which factor does ordinary LS inflate the variance here, and why is the WLS estimate so close to the radar reading?

## Problem 3.2 — Regularised least squares as prior-plus-measurement fusion

A tracker predicts range and range rate **m** = [42.0, −5.0]ᵀ with prior covariance **P** = [[1.0, 0.3], [0.3, 0.25]]. The radar of Problem 3.1 measures the range only, *y* = 41.6 m, **R** = 0.09 m², **G** = [1 0].

1. Compute **P**⁻¹, Λ_post = **G**ᵀ**R**⁻¹**G** + **P**⁻¹ and **P**⁺.
2. Compute x̂_ReLS = x̂_MAP.
3. Verify through the gain form **K** = **PG**ᵀ(**GPG**ᵀ + **R**)⁻¹, x̂ = **m** + **K**(y − **Gm**), **P**⁺ = (**I** − **KG**)**P**.
4. The range rate is not measured, yet it is corrected: explain why, and what happens when the off-diagonal entry of **P** is zero.
5. Discuss the limits **P**⁻¹ → 0 and **R**⁻¹ → 0.

## Problem 3.3 — Sequential least squares is batch least squares

Two measurements **y**₁ = **G**₁**x** + **r**₁, **y**₂ = **G**₂**x** + **r**₂ with independent noises **R**₁, **R**₂ and a prior (**m**, **P**₀).

1. Apply the information-form update twice and show that estimate and covariance coincide with the batch regularised estimate obtained from the stacked **G** and the block-diagonal **R**.
2. Deduce that the processing order is irrelevant.
3. Start from the radar of Problem 3.1 (x̂₁ = 41.6, P₁ = 0.09), process the camera reading with the gain form and verify numerically that the WLS values are recovered.
4. Identify the step that fails when **r**₁ and **r**₂ are correlated, and relate it to double counting in track-level fusion.

## Problem 3.4 — One Kalman-filter step by hand

State [p, v]ᵀ, **F** = [[1, T], [0, 1]] with T = 0.5 s, **Q** = σ_a² [[T⁴/4, T³/2], [T³/2, T²]] with σ_a = 0.8 m/s². Posterior at k−1: x̂ = [50.0, −4.0]ᵀ, **P** = diag(0.25, 0.16). Measurement y_k = 47.4 m, **H** = [1 0], **R** = 0.09 m².

1. Compute **Q**.
2. Compute the prediction (state and covariance).
3. Compute innovation, innovation covariance and gain.
4. Compute the posterior state and covariance; verify with the Joseph form.
5. Compute the NIS and compare with the two-sided 95 % interval of χ²₁, [0.00098, 5.02].
6. Only the position is measured, yet the velocity is corrected and the position standard deviation drops from 0.55 m to 0.26 m: explain both facts through the entries of **P**_{k|k−1}.

## Problem 3.5 — EKF Jacobian of a range–bearing sensor and azimuth wrapping

Measurement **g**(**x**) = [√(p_x² + p_y²), atan2(p_y, p_x)]ᵀ of **x** = [p_x, p_y, v_x, v_y]ᵀ, σ_θ = 0.5°.

1. Derive the Jacobian and compare with the first two rows of the radar Jacobian of Example 3.1.
2. Evaluate it at x̂ = [40, 3, −5, 0]ᵀ.
3. The bearing row scales as 1/d: consequence for the lateral information; lateral standard deviation implied by σ_θ at 60 m and 10 m.
4. Predicted bearing 3.135 rad, measured −3.140 rad: raw difference, wrapped innovation, NIS in both cases with S ≈ σ_θ², and the effect of a χ²₁ gate at 5.02.
5. Why must the wrapping be applied to the innovation rather than to the predicted measurement or the state?

## Problem 3.6 — Zonotope propagation and a single-strip correction

X_k = p ⊕ H B∞² with p = [1, 0]ᵀ, H = [[1, 0.5], [0, 1]]; dynamics F = [[1, 1], [0, 1]] (T = 1 s) with bounded acceleration |a| ≤ ā = 0.2, mapped onto the generator ā [T²/2, T]ᵀ.

1. Predicted zonotope (centre, three generators) and its interval hull.
2. Position measured as y = 1.4 with |Δy| ≤ φ̄ = 0.3: write the strip with normal c = [1, 0]ᵀ.
3. Gain λ* = H̃H̃ᵀc / (cᵀH̃H̃ᵀc + φ̄²), corrected centre, corrected generator matrix (four columns), interval hull.
4. Repeat with λ = c; compare the interval hulls and ‖Ĥ‖_F²; which quantity does each choice minimise?
5. Exact interval hull of the intersection (two LPs over ξ ∈ [−1, 1]³) and comment on the conservatism of both enclosures.

## Problem 3.7 — Dempster's rule versus Smets' rule

Ω = {P, V, S}. Camera: m_C(P) = 0.5, m_C({P,V}) = 0.3, m_C(Ω) = 0.2. LiDAR: m_L(V) = 0.4, m_L(S) = 0.1, m_L({P,V}) = 0.4, m_L(Ω) = 0.1.

1. Conflict mass K.
2. Fused BBA under Dempster's rule.
3. Bel, Pl and pignistic probability of each singleton.
4. Fused BBA under Smets' rule, including m(∅); what do the two rules do differently with the same products?
5. Sources m₁(P) = 0.99, m₁(S) = 0.01 and m₂(V) = 0.99, m₂(S) = 0.01: K and both combinations; why is the Dempster result unsafe for a planner?

## Problem 3.8 — Covariance intersection

Two consistent tracks (x̂₁, **P**₁), (x̂₂, **P**₂) with unknown cross-covariance.

1. Prove that the CI estimate is consistent for every ω ∈ [0, 1] and every admissible cross-covariance (hint: bound the joint covariance by the block-diagonal matrix diag(P̄₁/ω, P̄₂/(1−ω))).
2. For **P**₁ = diag(0.25, 1.0) and **P**₂ = diag(1.0, 0.16): closed form of tr **P**_CI(ω), the trace-optimal ω*, and **P**_CI(ω*).
3. Compare with the independent-track fusion covariance and explain why CI stays direction dependent although ω is a scalar.

## Problem 3.9 — Incomplete AHP by logarithmic least squares

Four criteria; comparisons A₁₂ = 3, A₂₃ = 2, A₁₃ = 4, A₃₄ = 2 only.

1. Comparison graph: connected? Is the cycle 1–2–3 consistent?
2. Show that the LLS normal equations read **L y** = **b** with **L** the graph Laplacian and b_i = Σ_j ln A_ij over the neighbours (A_ji = 1/A_ij); write **L** and **b**.
3. Solve with the gauge Σ y_i = 0 and normalise.
4. Fitted ratios and residuals on the four edges; how is ln(A₁₂A₂₃/A₁₃) distributed, and why is edge (3,4) fitted exactly?
5. Predict A₁₄ and A₂₄.
6. What changes if A₁₃ is removed?

## Problem 3.10 — Computational: process noise and innovation consistency

Reproduce the radar cut-in scenario of Example 3.1 (ego 27 m/s, target 22 m/s initially 60 m ahead in the adjacent lane, lane change from 3.5 m to 0 m between t = 5 s and t = 8 s, radar at 20 Hz with σ_r = 0.30 m, σ_θ = 0.5°, σ_ṙ = 0.10 m/s) and run the constant-velocity EKF on one and the same detection sequence for σ_a ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4} m/s². Report position RMSE, mean NIS and the fraction of samples inside the 95 % interval of χ²₃, separately on the constant-velocity segments and inside the lane-change window.

1. For which values is the filter overconfident, for which underconfident? Use the NIS, not the RMSE.
2. Why does the constant-velocity RMSE first decrease and then increase with σ_a, whereas the lane-change RMSE decreases monotonically?
3. Is there a σ_a for which the NIS is consistent in both segments? Relate the answer to interacting multiple-model filters.

Run: `python3 exercises/ch03/solutions/ch03_prob10_ekf_process_noise.py` (from the repository root).
