# Headline results

Every number below is checked mechanically by `python repro/check_numbers.py` against the
pinned dumps in `repro/`. Paper tables: Table 2 (opponent matrix), Table 3 (FR3 arm),
Table 4 (MUKCa sockets), Table 5 (ABB range-only); details in the appendices.

## Synthetic composition (opponent matrix, E1)

| Model | Parameters | Geodesic error |
|---|---|---|
| **LawKAN-SU(2), K=1** | **54** | **2.78°** |
| MLP + quaternion head h=32 | 1476 | 3.36° |
| MLP + 6D head | 1542 | 3.91° |
| MLP + 9D-SVD head | 569 | 4.32° |
| Measured Bayes floor (matched E1 draws) | — | 2.733° |

The 54-parameter dense LawKAN edge is the most accurate dense model with 27× fewer parameters than the strongest matrix opponent
(92× fewer than the strongest tuned opponent, MLP-Quat h=64 at 3.25°); the pruned
18-parameter edge is 2.75°. It sits 0.05° above the matched Bayes floor (+0.63° for the strongest matrix opponent, +0.52° tuned). Sample efficiency
is ≈32× versus the fixed-protocol 6D opponent and ≈7× versus the strongest tuned one (E39).

## Real arm calibration (FR3, 114 poses) — where the law is deformed

Isolated zero-initialized correction ψ, final two-slot head (306 parameters), frozen exact edge:

| Probe | LawKAN ψ | Same-class reference | Note |
|---|---|---|---|
| FR3, N=20 | **1.51 mm** | GN-LS 1.56 mm | two-slot head no longer overfits (X35) |
| FR3, N=40 | **1.24 mm** | GN-LS 1.39 mm | significant gain, p=0.040 |
| FR3, N=60 | **1.14 mm** | GN-LS 1.29 mm | 5-seed protocol (X35) |
| FR3, N=80 | **1.12 mm** | GN-LS 1.33 mm | 5-seed protocol (X35) |

Exact-node floor without ψ: 1.37/1.40 mm at N=60/80; GN is 1.29/1.33 mm. The correction is
significantly better at N=40 (p=0.040) and N=80 (p=0.002), within noise at N=20/60. The
recount under the final two-slot head is X35; the previous (pre-final Fourier) cells were
2.49/1.50/1.17/1.14 mm at N=20/40/60/80. The 10-seed replication campaign (P-D) is pinned as `g2bc`.

**Strong learned baseline (X22).** A residual MLP selected on a validation split
(h=128/256; 18k–69k parameters, 59–225× more than the law model's 306) is significantly
worse at small N (+0.88 mm at N=20, p=0.005; +0.79 mm at N=40, p=0.049); at N=60/80 the
paired comparison is not significant (1.35/1.33 vs 1.14/1.12 mm), and even its test-set
oracle (1.27/1.26 mm) stays above the correction. The advantage is parameter and data
efficiency, not accuracy in the large-N limit.

## Socket and range-only replication (Tables 4–5)

MUKCa sockets (Table 4; KUKA/Panda/Kinova 10 seeds, FR3 unit 5 seeds; held-out RMS, mm):

| Platform | Dense screw | Reduced edge (ψ=0) | Two-slot ψ |
|---|---|---|---|
| KUKA iiwa14 | 0.26 | 0.55 | **0.166** |
| Panda | 0.49 | 0.93 | **0.330** |
| Kinova Gen3 lite | 1.98 | 2.47 | 2.325 (kernel **1.908**) |
| FR3 unit (boundary) | **0.29** | 1.49 | 0.441 |

ABB RobotCali draw-wire (Table 5; held-out mean |ΔL|, mm; global ψ vs GN):

| N | GN | ψ=0 (Adam) | Global ψ | p |
|---|---|---|---|---|
| 80 | 1.41 | 1.53 | **1.37** | 0.74 |
| 160 | 1.20 | 1.38 | **1.11** | 0.013 |
| 320 | 1.14 | 1.33 | **1.06** | 0.0004 |

*The socket two-slot head is the final recount (X36); on ABB the global head is best (it beats
GN at N≥160, ties at N=80) and the local kernel adds nothing. On the socket the global slot
carries the signal except on Kinova, where the local kernel is the collapse-safe choice (the
two-slot sums both). The FR3 unit is an attribution boundary: the dense screw class wins.*

Pooled registered criterion (Fisher over ABB N=80 and KUKA, 10 seeds): χ²=58.5,
p ≈ 6·10⁻¹²; extended with Kinova under the local-kernel head (χ²=78.1, p ≈ 9·10⁻¹⁵) — the
claim is made across three platforms. Kinova needs the local-kernel head: it removes the
global-slot seed collapse, beats the reduced edge (p<10⁻⁴), and matches the dense screw
(p=0.101, ns).

## Revision round and pole-fix branch (X22–X61)

The isolated correction's seed instability is localized and fixed. Two mechanism hypotheses
are refuted and reported: the exponential-winding hypothesis (`X25`/`X25b`) and the
unbounded-magnitude hypothesis (`X26`). The remedy is **capacity** — an L2 weight-decay window
on the ψ head (`X26b`), with the optimal λ* falling as training size grows (`X26c`: 3e-3 → 0) —
and **locality** — a local RBF-kernel head removes the Kinova collapse without L2 (`X32`) and
improves the main FR3 result at small N (`X34`: −0.86/−0.20 mm at N=20/40, p=0.016/0.026). The
recommended parameterization is a **two-slot** head (global linear trend + local kernel), which
dominates both pure heads at every training size (`X33`); the global-slot weight grows with N.
The pole-fix branch (`X37`–`X61`) then closes the residual training-spike question. The input
chart crossing π is the operand singularity (`X53-print-2`). Quaternion features halve the
peak but cost final error (`X53-fix`), and a group-metric kernel does not recover the final
error (`X60`). Polyak/EMA removes the spikes in every configuration while leaving the final
error unchanged (`X59`, `X61`). The branch-safe global-only variant shows no measured cost at the
tested budget, though equivalence at the ±10% margin was not established (`X61`).

## Parity on real sensors and robots

- Gyroscope calibration: a 12-parameter learned fit matches the classical
  Levenberg–Marquardt fit of the same model to ≤ 0.009°, and gains over bias-only
  subtraction at 0.35 s.
- Real IMU dead-reckoning: the pruned 18-parameter edge matches the training-free
  integrator (1.057/1.189° at 1 s / 5 s vs 1.148/1.191°) and is seed-deterministic where
  the dense edge is bimodal.
- Three-platform FK calibration (FR3/UR10/UR5) and the ABB/KUKA/Panda socket protocols:
  parity where the classical estimator already carries the same law, with data efficiency
  at small samples.

## Where it loses

- No group law (null task): a plain MLP wins.
- Hidden rotation angle unrelated to any input (information-limited).
- Dense IMU edge at long horizons (seed-bimodal); the pruned edge is the deterministic one.
- Kinova (socket): the global head is seed-unstable; the local kernel removes the collapse,
  beats the reduced edge, and matches the dense screw (ns) rather than beating it.

Refuted and borderline predictions are reported: **130 registered verdicts** (59 main battery +
30 validity package + 30 revision round + 11 pole-fix branch); see the supplementary prediction
log and the paper's verdict appendix.
The full per-entry log (prediction → outcome → verdict, with the refuted entries kept) is in
[`predictions.md`](predictions.md).
