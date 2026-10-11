# Chapter 7 exercises: Applications Beyond Autonomous Vehicles

Problems 1-5 are analytical (solutions in the book's back matter). Problems 6 and 7 are
computational; reference solutions are in `solutions/`.

1. **Evidence combination by hand.** AIS cue m1({N})=0.7, m1(Ω)=0.3; radar cue m2({T})=0.4,
   m2({L,T})=0.4, m2(Ω)=0.2 on Ω={N,L,T}. Compute the conflict K, the Dempster-combined BBA,
   Bel(T), Pl(T) and the pignistic probabilities. Discount the AIS source with α=0.5 and repeat.
2. **Windowed chi-square threshold.** n_y=4 innovation components per 10-s period, W=12. Derive
   the H0 distribution of the windowed statistic, its one-sided 99% threshold and two-sided 99%
   interval (chi-square table or Wilson-Hilferty). Explain why overlapping windows make the alarm
   rate differ from α and bound the expected number of alarm samples per hour.
3. **Non-central NIS.** Show that a constant innovation bias b gives a non-central chi-square NIS with
   λ = bᵀS⁻¹b and a windowed statistic with non-centrality Wλ. For sqrt(S11)=31.3 m, n_y=2, W=12
   and a 50-m bias, compare the means with the 99% thresholds.
4. **MCDM scores and rank reversal.** Verify the four alert scores of the book example and find the
   proximity weight at which alert a3 overtakes a1 (rescaling the other weights proportionally),
   or show that none exists.
5. **DAS measurement matrix.** Write H_jn for a channel at 250 m (L=1000 m, L_g=10 m, n=1..4), show
   it is blind to mode 2, and determine which modes a channel at 500 m cannot observe.
6. **(Computational) Spoofing ramp and coloured noise sweeps.** Vary a_s ∈ {0.0005, 0.001, 0.002,
   0.004} m/s² and W ∈ {6, 12, 24}; tabulate detection delays and the AIS offset at detection.
   Repeat the coloured regime with φ ∈ {0.5, 0.7, 0.85, 0.95} and report the Ljung-Box alarm
   fraction. Solution: `solutions/prob06_spoof_sweep.py`.
7. **(Computational) Strike position, number of modes, coupling artefact.** Move the strike to
   x_e ∈ {200, 400, 640, 800} m with n_m ∈ {3, 5, 8}; report detection delay and centroid error.
   Then let the filter assume κ=1 on the two free-span channels (true 0.5) and describe the residual
   profile. Solution: `solutions/prob07_cable_sweep.py`.
