# Chapter 5 — Autonomous Driving as a Representative Application: problems

Problems 5.1–5.8 of the book (section "Problems" at the end of Chapter 5). Solutions are in the
back matter of the book; the scripts for the computational problems are in `solutions/`.
Worked examples referenced below live in `companion_repo/examples/ch05_gnss_imu_odometry_ekf/`
and `companion_repo/examples/ch05_camera_radar_ci_fusion/`.

## 5.1 Calibration errors (analytical)
A forward camera has focal length f_x = f_y = 1200 px and a 1920×1080 image. Its extrinsic calibration
with respect to the LiDAR has a yaw error of 0.5°.
(a) By how many pixels is a projected LiDAR point displaced horizontally, and does it depend on range?
(b) Lateral position error produced by the same yaw error for an object at 20 m and at 80 m when the
camera bearing is used to position it.
(c) A translation error of 10 cm along the camera x axis instead: pixel error at 20 m and 80 m; what does
this imply for near-field versus far-field fusion?

## 5.2 Time offset (analytical)
Camera and radar have a residual time offset of 30 ms; the ego vehicle drives at 20 m/s.
(a) Apparent longitudinal misalignment for a static obstacle and for an oncoming vehicle at 20 m/s.
(b) With a fused longitudinal standard deviation of 0.3 m, compute the normalised distance in both cases
and decide whether a gate with γ = 9.21 (99 %, 2 d.o.f.) would still associate the detections.
(c) Propose a state augmentation to estimate the offset online and state when it is observable.

## 5.3 Dead-reckoning drift (analytical)
Straight driving at speed v with an uncompensated gyroscope bias b.
(a) Show that the heading error after τ is bτ and the cross-track error ≈ v b τ²/2.
(b) Evaluate for b = 10 mrad/s, v = 14 m/s, τ = 25 s; compare with the 63.9 m of the three-state filter
of the worked example and explain the difference.
(c) Add a wheel scale error s = 2 %: along-track error after 25 s, total error, and which error a
tactical-grade IMU would reduce.

## 5.4 GNSS outage variants (computational) — `solutions/prob04_gnss_outage_variants.py`
Re-run the GNSS–IMU–wheel-odometry filter with (a) a 60 s outage from t = 50 s and (b) 1 Hz GNSS fixes
before a 25 s outage. Report the position error at the end of the outage, the reported 3σ radius and
whether the error is inside it, and the 3σ uncertainty of the gyro-bias estimate at the tunnel entrance.

## 5.5 Identical covariances (analytical)
P1 = P2 = P. (a) Show that independent fusion gives P/2 while CI gives P for every ω.
(b) Interpret. (c) With P12 = ρP, compute the actual covariance of the naive fused estimate and its
expected NEES for a 2-D state, and the ρ above which the naive rule under-reports its variance by more
than 50 %.

## 5.6 CI sensitivity (computational) — `solutions/prob06_ci_sensitivity.py`
Vary the camera range accuracy σ_r/r ∈ {5, 15, 30} % and the radar bearing accuracy
σ_θ ∈ {0.5, 1.5, 3}°; report ω*, tr P_CI and tr P_f (naive) for the nine combinations and explain why ω*
changes so little. Then use an imaging radar with σ_θ = 0.1° and show what happens when one track
dominates the other in every direction.

## 5.7 Metrics (analytical)
(a) NDS from mAP = 0.60, mATE = 0.30, mASE = 0.25, mAOE = 0.40, mAVE = 0.35, mAAE = 0.15; then with
mAVE = 0.20. (b) MOTA for 1000 GT objects, 120 misses, 80 false alarms, 10 identity switches.
(c) ECE for two bins: 400 predictions (conf 0.86, acc 0.74) and 600 predictions (conf 0.95, acc 0.91).

## 5.8 Latency budget (analytical)
τ_sense = 33 ms, τ_xfer = 5 ms, τ_detect = 40 ms, τ_fuse = 5 ms, τ_plan = 60 ms, τ_ctrl = 10 ms,
τ_act = 150 ms. (a) End-to-end latency and distance at 30 m/s. (b) Ages of the camera detection and of a
radar track (15 ms): to which time should both be predicted, and how much does the position uncertainty
grow over the larger age for σ_a = 0.8 m/s²? (c) Effect of pipelining the detector on two GPUs; which
terms must shrink to reduce the stopping distance?

Run the scripts with the system `python3` (numpy only):

```
python3 solutions/prob04_gnss_outage_variants.py
python3 solutions/prob06_ci_sensitivity.py
```
