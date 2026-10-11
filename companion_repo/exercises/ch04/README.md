# Chapter 4 — Learning-Based and Hybrid Sensor Fusion Techniques: problems

Solutions scripts for the computational problems are in `solutions/` (run with the system `python3`; only numpy is needed).
The worked examples of the chapter are in `companion_repo/examples/ch04_hybrid_hetero_kf/` and `companion_repo/examples/ch04_pose_graph_1d/`.

1. **Mixture-of-experts gate.** Three range estimates x̂ = (41.8, 42.6, 43.1) m are gated by g_i ∝ exp(β q_i) with quality indicators q = (0.3, 0.5, 0.9). Compute the gate and the fused estimate for β = 0, 4, 20, and show analytically that the fused estimate tends to the highest-quality source as β → ∞. Compare with the proportional weights w_i = q_i / Σ_j q_j.

2. **Learned reliability in weighted least squares.** Repeat the logistic-reliability example of the chapter (p_i = σ(−2 + 5 q_i), nominal σ = (1.0, 0.5, 0.8) m) for a clear-day scene with q = (0.9, 0.8, 0.9). Give the reliabilities, the fused estimate and its standard deviation, and explain why the result is now close to plain weighted least squares.

3. **Calibration.** Five confidence bins with counts (100, 200, 300, 250, 150), mean confidences (0.30, 0.50, 0.70, 0.90, 0.98) and accuracies (0.35, 0.45, 0.60, 0.80, 0.90): compute the expected calibration error. For logits z = (3, 1, 0) compute the softmax at temperatures T = 1 and T = 2 and verify the arg-max is unchanged.

4. **Log-odds occupancy.** With p_hit = 0.7, p_miss = 0.4, prior L_0 = 0 and clamping to [−2, 3.5], a cell receives 6 hits and then n free observations. After how many free observations does P(occupied) fall below 0.5, with and without clamping? Comment on clamping for dynamic objects.

5. **Pose graph by hand.** Redo the 1-D four-pose example when the loop closure x_3 − x_0 = 2.7 m has standard deviation 0.05 m (odometry 0.10 m). Derive the normal equations, solution, residuals, and the standard deviation of x̂_3. Show that for n equally weighted odometry steps and a loop closure with weight ratio ρ = σ_odo² / σ_lc², the loop-closure residual is the discrepancy divided by 1 + nρ.

6. **Computational — miscalibrated uncertainty head.** Re-run the heteroscedastic-detector tracker with the learned standard deviation scaled by c ∈ {0.5, 1, 2}; report RMSE, ANIS and the fraction of NIS values inside the 95 % χ²₂ band. Which case is over-/under-confident, and why can RMSE alone not detect the miscalibration? Script: `solutions/prob6_miscalibrated_head.py`.

7. **Computational — planar pose graph.** Four SE(2) poses around a 10 m square with noisy odometry (0.3 m, 0.3 m, 2°) and one loop closure (0.1 m, 0.1 m, 1°). Implement Gauss–Newton with the first pose anchored, starting from dead reckoning; report the last-pose position error before/after, the weighted cost, and the marginal standard deviations of the last pose. Script: `solutions/prob7_pose_graph_2d.py`.

8. **Fusion operators.** Show that concatenation + linear projection can represent any fixed gate but not an input-dependent one. Compute the cross-attention output for d_k = 2, query (0, 1), keys (1, 0), (0, 1), (−1, 0), values 2, 0, −2, and compare with the chapter's example (query (1, 0)).
