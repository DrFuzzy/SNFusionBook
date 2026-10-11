# Chapter 8 — Open Challenges and Future Directions: exercises

Problems 1, 4, 5 and 6 are design/discussion questions whose model answers are in the book's
"Solutions to Selected Problems". Problems 2 and 3 are computational; a solution script and the
corresponding notebook are provided in `solutions/`.

| Problem | Type | Script / notebook |
|---|---|---|
| 8.1 Sensor-health monitor | design | — |
| 8.2 Expected calibration error | computational | `solutions/ch08_p2_ece.py`, `.ipynb` |
| 8.3 Timestamp offset and the innovation test | computational | `solutions/ch08_p3_timestamp_offset.py`, `.ipynb` |
| 8.4 Sim-to-real transfer protocol | design | — |
| 8.5 Cooperative perception message | discussion | — |
| 8.6 A foundation model as an information source | discussion | — |

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib`. Run from this folder, e.g.
`python3 solutions/ch08_p2_ece.py`. Each script prints the numbers quoted in the solutions and
saves a figure next to it. Notebooks were generated with `tools/py2nb.py` from the scripts.

## Problem statements

**8.1 Sensor-health monitor (design).** A vehicle fuses camera, radar and LiDAR detections in a
Kalman tracker (hybrid pipeline of Chapter 4). Design a sensor-health monitor that uses the
per-sensor NIS, an evidence-theoretic discount factor per sensor and a graceful-degradation
policy. Specify (a) the statistics computed and over which window, (b) the thresholds and how
their false-alarm rate is set, (c) how the discount factor is derived from the statistics, and
(d) how the policy distinguishes a blocked sensor (no detections) from a miscalibrated one
(biased detections) and what the planner is told in each case.

**8.2 Expected calibration error (computational).** A detector reports twelve confidences
`p = [0.95, 0.92, 0.88, 0.85, 0.78, 0.72, 0.65, 0.58, 0.52, 0.45, 0.35, 0.22]` with correctness
`y = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]`. (a) Compute the ECE with M = 5 equal-width bins on
[0, 1]. (b) Compute the Brier score, the accuracy and the mean confidence. (c) Is the detector
over- or under-confident? Temperature scaling replaces p_i by sigma(z_i / T) with
z_i = logit(p_i): is T > 1 or T < 1 needed? Verify numerically by finding the T that minimises
the negative log-likelihood. (d) Comment on the reliability of an ECE estimate from twelve samples.

*Expected output:* ECE = 0.2275, Brier = 0.2313, accuracy 0.50, mean confidence 0.656,
T* ≈ 1.82 (over-confident, T > 1), ECE after scaling ≈ 0.13.

**8.3 Timestamp offset and the innovation test (computational).** A radar measures the range to
a lead vehicle with sigma = 0.3 m; the innovation covariance is S = sigma^2. The radar clock is
offset by delta so that each measurement is interpreted delta seconds late; the lead vehicle
closes at v = 15 m/s. (a) Show that the offset produces an innovation bias b = v * delta and
compute it for delta = 20 ms and 50 ms. (b) Show that the expected NIS becomes 1 + b^2/sigma^2
and evaluate it. (c) The single-sample 95 % gate for chi-square(1) is 3.84: for which offset is
a correct measurement rejected on average? (d) A windowed test averages the NIS over N = 20
samples and compares it with chi-square(20, 0.95)/20 ≈ 1.57: find the smallest detectable delta
at v = 15 m/s and at v = 2 m/s. (e) Interpret the result for urban versus motorway driving and
propose how delta could be estimated rather than merely detected.

*Expected output:* b = 0.30 m / 0.75 m; E[NIS] = 2.0 / 7.25; rejection probability 0.17 / 0.70
(0.05 without offset); minimum detectable offset 15 ms at 15 m/s and 113 ms at 2 m/s.

**8.4 Sim-to-real transfer protocol (design).** A camera–LiDAR detector with learned covariances
was trained entirely in simulation. Propose a protocol, using only unlabelled real recordings,
to decide modality by modality whether the detector and its uncertainty model transfer. Specify
the statistics (NIS of the tracker fed by the detector, conflict mass between the camera and
LiDAR branches, temporal consistency of detections), the reference values expected if transfer
were perfect, and accept/reject thresholds. Discuss what a small labelled subset would add.

**8.5 Cooperative perception message (discussion).** A roadside unit broadcasts object tracks to
approaching vehicles. (a) List the minimal fields a message must carry for the receiver to apply
covariance intersection and evidence discounting consistently. (b) Explain what goes wrong if
the covariance is omitted and the receiver assumes a default one, and if the same track is
received through two paths without provenance. (c) Propose how the receiver could estimate its
trust in the sender from the consistency of past messages and how that trust should enter the
fusion.

**8.6 A foundation model as an information source (discussion).** A multimodal language model
labels rare objects in camera images with a free-text class and a self-reported confidence.
Explain how its output can be mapped to a basic belief assignment, how a discount factor can be
estimated from a held-out set, how the conflict mass K with the explicit sensors should be
interpreted, and where in the hybrid pipeline such a source should be inserted so that it
cannot override a consistent classical estimate.
