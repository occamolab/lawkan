# Block and script map

Script-by-script map from this repository to the paper, plus the per-block details and the
frozen sanity numbers. Regenerate the mechanical tables for every block with
`python repro/make_tables.py`.

## Repository layout

Every headline experiment maps to a script (a few derived counts are computed from several
dumps): the core battery lives in
`experiments/synthetic/`, the real-data probes and the validity package in
`experiments/real/`. The synthetic-core scripts share one protocol (800 train /
4000 test, shared seeds, geodesic loss, paired t-tests); the real-data probes
document their own protocol inline. Experiment scripts end with a machine-readable **SUMMARY-JSON** dump,
from which the paper tables are generated mechanically; helper and figure scripts read those dumps.

The per-entry **registered-predictions log** (prediction → outcome → verdict, refuted entries kept) is
[`predictions.md`](predictions.md).

| File | Contents | Paper |
|---|---|---|
| `run_main_experiments.py` | Main synthetic benchmark, blocks E1–E14 (~820 lines) | §4.2.1; App. C (exactness), D (extended), E (compact), K (predictions) |
| `run_extensions.py` | Applied synthetic probes + extra opponents: E15 (IMU), E17 (rigid-body feed-forward), E18 (BCH), X1–X4 (RPMG, non-Gaussian, Vector-Neurons) | §4–§5; App. D.4, F, G |
| `E16_gl2_composition.py` | GL⁺(2,ℝ) probe (self-contained, own generator + `matrix_exp` readout) | §4.1; App. D.1 |
| `E19_sparse_edge.py` | Sparse-edge battery E19.0–E19.8 (Hypothesis N; P23–P27) | §4.2.1; App. K |
| `E19b_random_mask.py` | Random equal-size 18-p mask control for the sparse>dense claim (E19b; P63) | App. H.3; App. K |
| `E20_higher_harmonic.py` | Order-rule demonstration (E20) | §4.2.3; App. I.3 |
| `E21_gl2_support_discovery.py` | Analytic-support discovery, GL⁺(2,ℝ) (E21) | App. D.1 |
| `E22_se3_composition.py` | SE(3) full design (E22) | §4.1; App. D.2 |
| `E23_se3_feature_ablation.py` | SE(3) feature-group blind test (E23) | §4.1; App. D.2, D.4 |
| `E24_order_k_scaling.py` | Order-$K$ scaling, blind held-out seeds (E24) | §4.2.3; App. I.3 |
| `E25_state_rigid.py` | State-conditioned edge on rigid-body dynamics (E25; P41) | App. G.2, K |
| `E25b_ff_poly_control.py` | Feed-forward one-pass polynomial control (E25b; P42) | App. G.2, K |
| `E29_feature_span.py` | Feature-span / tensor-extension at variable angles (E29; P45) | §4.2.3; App. I.1; App. K |
| `E30_full_so3.py` | E1 opponent matrix on a full SO(3) readout (E30) | §4.2.1; App. D.4 |
| `E31_node_isolation.py` | Node/edge isolation: fixed exp+Hamilton node, KAN/MLP/linear edges vs no-node controls (E31; P49) | §4.2.2; App. K |
| `E31b_node_mlp_h4.py` | Edge-capacity bracket: MLP h=4 (1L/2L) vs h=8 under the same node (E31b; P51) | §4.2.2; App. K |
| `E31c_sparse_edge.py` | Analytic-support (sparse) 18-p edge under the E31 protocol (E31c; P52) | §4.2.2; App. K |
| `E31d_pruning_curve.py` | Automatic pruning of the K=2 feature dictionary under the E31 node: train -> rank features per group -> keep top-m -> retrain (E31d; P54) | §4.2.2; App. K |
| `E32_opponent_tuning.py` | Opponent tuning grid h×lr, best cell (E32; P50) | §4.2.1; Figure A2; App. K |
| `E33_floor_reported_metric.py` | Bayes floor as minimizer of the reported metric (E33; corrects E10/E10b) | §4.2.1; App. D.4 |
| `E33b_metric_loss.py` | Train on the reported metric itself (`mean acos`); refuted (E33b; P53) | §4.2.1; App. D.4, K |
| `E33c_metric_loss_stable.py` | Numerically stable metric loss `2 asin(||d||/2)` (E33c; P55, after P53) | §4.2.1; App. D.4, K |
| `E36_e30_mc_floor.py` | Monte-Carlo SO(3) floor for E30 with input noise (E36; closes audit C4) | §4.2.1; App. D.4 |
| `E38_tuned_opponents.py` | Fairness audit: tuned opponents and per-edge learning rates under the E1/E31 protocol (E38) | §4.2.1; App. K |
| `E39_sample_efficiency_tuned.py` | Sample-efficiency curves for the tuned opponents (E39; fairness audit of the 32x claim) | §4.2.1; Figure A3; App. K |
| `E40_floor_matched.py` | Matched Bayes floor evaluated on the same E1 test draws (E40) | §4.2.1; App. D.4 |
| `E34_x7_recipe.py` | Section-3/5 recipe on the real arm: per-joint theta-dependent edge + E21-style pruning (E34; P56–P59). **Journal-only probe, not part of the paper's prediction log** | — |
| `X5_broad_real_imu.py` | BROAD real-IMU orientation probe, 0.07–5 s windows, sparse variant | §5.5.1; App. F.1 |
| `X6_broad_calibration.py` | BROAD gyroscope calibration, learned (M, b), 12 parameters | §5.5.2; App. F.2 |
| `X7_robot_calibration.py` | Real Franka kinematic calibration (114 poses), per-joint SE(3) correction vs PoE-LS / MLP-residual (earlier bench protocol) | §5.1 |
| `X8_broad_seq.py` | Sequential/residual baselines on BROAD (GRU, ResidMLP; P30–P31); requires `X5_broad_real_imu.py` alongside (import) | §5.5.1; App. F.1 |
| `X9_imu_tuning_sensitivity.py` | IMU learning-rate / tuning-sensitivity sweep | §5.5.1; App. F.1 |
| `X10_fk_calib_real.py` | FK calibration on three real platforms (FR3/UR10/UR5), screw-parameterized exact SE(3) composition vs constant base/tool fit (X10; P48) | App. F.3; App. K |
| `E26b_baxter_dynamics.py` | Real Baxter arm dynamics audit: state-conditioned edges + exact integrator (E26b) | App. G.3 |
| `E26c_error_law.py` | Deterministic error-accumulation law on the edge, real Baxter (E26c) | App. G.3 |
| `S1_synthetic_law.py` | Synthetic controls: exact vs deformed law (S1a/S1b) | App. G |
| `S2_ur5gtpn_relaxed.py` | Relaxed law on the independent UR5 dataset | App. G |
| `S3_matched_mlp.py` | Parameter-matched residual-MLP control | §5.1 |
| `S8_ur10_relaxed.py` | Relaxed law on UR10 (identifiability failure mode) | App. G |
| `S1b_synthetic_law_twoslot.py` | S1 synthetic law controls under the final two-slot head | App. G |
| `S2b_ur5_twoslot.py` | S2 UR5-GTPN relaxed law under the final two-slot head | App. G |
| `S8b_ur10_twoslot.py` | S8 UR10 relaxed law under the final two-slot head | App. G |
| `G2a_ur10_isolated.py` | Isolated law correction on UR10 (reads the X10b GN baseline) | App. G |
| `G2bc_isolated_psi.py` | Isolated law correction: FR3/UR5/synthetic; 10-seed P-D run and cluster split | §5.1; App. G |
| `G3_orthogonality.py` | Correction-vs-edge orthogonality decomposition | App. G |
| `G3b_orthogonality_twoslot.py` | G3 orthogonality decomposition under the final two-slot head | App. G |
| `G4_gauge_compare.py` | Isolation vs sparse vs projected correction (FR3) | App. G |
| `X6b_tinygc_style.py` | TinyGC-style 27-p calibration subnet, LM parity (P-X6b) | App. F |
| `X7_vqf_reference.py` | VQF classical AHRS reference on BROAD (disturbed/undisturbed) | App. F |
| `X11_ur5_gtpn_sweep.py` | Independent UR5-GTPN sweep: const/θ edges, Gauss–Newton | App. G.4; Figure A9(c) |
| `X12_chain_errlaw.py` | Chain error law, shared sources on FR3 | App. G.4 |
| `X12b_dense_chain.py` | Per-joint sources + chain coupling (FR3) | App. G.4 |
| `X13_learned_node.py` | Learnable node / law relaxation on FR3 | App. G.4 |
| `X14_fourier_law.py` | Fourier law correction (KAN head), FR3 | §5.1; App. G |
| `X16_psi_diagnostics.py` | Learned-law diagnostics: channel groups, input ablation, partial dependence | App. G.4; Figure 3(b) |
| `X16b_psi_diagnostics_twoslot.py` | X16 diagnostics under the final two-slot head | App. G.4 |
| `X17_sparse_law.py` | Sparse law correction (top-k chart atoms, FR3) | App. G.4 |
| `X18_coning.py` | Coning probe on the tiny gyroscope (P-CON1–P-CON3) | App. G.4 |
| `X18b_coning_constrained.py` | Constrained coning correction (P-CON2b) | App. G.4 |
| `X19_tau_psi.py` | Gravity-torque physics ψ on FR3 (P-TAU) | App. G.4 |
| `X19_recover.py` | Recovers the P-TAU verdicts from the archived training log + canonical g2bc dump | App. G.4 |
| `X20_fric_synth.py` | Synthetic backlash signature (P-FR1–P-FR3) | App. G.4 |
| `X21_fric_proxy.py` | IPA backlash proxy (P-FRIC-proxy) | App. G.4 |
| `MK1_mukca.py` | MUKCa socket consistency: KUKA/Panda/FR3-unit/Kinova (P-MK1–P-MK7) | §5.3; App. F.3; Table 4 |
| `MK1b_mukca_nominal.py` | MUKCa nominal-URDF forward kinematics helper (shared by MK1/diagnostics) | — |
| `MK1c_shuffle_control.py` | Shuffled-label control (P-MK6) | App. F |
| `RC1_robotcali.py` | ABB RobotCali draw-wire range-only calibration (P-RC1–P-RC4); **external data** | §5.4; App. F.3; Table 5 |
| `POOL_verdict.py` | Pooled cross-platform verdict (P-POOL / P-POOL-EXT) | App. F |
| `POOL2_verdict.py` | Pooled cross-platform verdict under the final heads (registered recompute) | App. F |
| `X10b_poe_ls_baseline.py` | Classical trust-region LS (GN) baseline on the X10 splits; read by S8/G2a | App. G (reference) |
| `X7_gn_ls.py` | Gauss–Newton PoE-LS solver shared by X7/X13/X14/G2bc | — (helper) |
| `X7_bench_main.py` | X7 bench rows for Table 3 and Figure 3 (exact/GN) | Table 3; Figure 3 (fig13) |
| `flops_convention.py` | Single-source FLOP accounting (flat convention); used by E2 and E19.8 | §3; App. B (FLOP accounting) |
| `X22_learned_baseline.py` | Strong tuned residual-MLP learned baseline for the FR3 law calibration (X22; P-LB1–P-LB3) | §5.1 (Table 3); App. K |
| `X24_fr3_gate.py` | Validation-gated checkpoint on the FR3 isolated correction (X24; P-FRG1–P-FRG3) | §5.1, §5.3; App. G; App. K |
| `X25_psi_bound.py` | Seed instability of the learned law: rotation-bounded chart, mechanism check (X25; P-B1–P-B3) | §5.1, §5.3; App. G; App. K |
| `X25b_weight_corr.py` | Cross-seed consistency of the learned psi function (X25b; P-B5) | §5.1, §5.3; App. K |
| `X26_fullbound.py` | Bounds every component of psi + rotation/translation decomposition of the collapse (X26; P-FB1–P-FB3) | §5.1, §5.3; App. G; App. K |
| `X26b_lambda.py` | L2 (weight-decay) window on the psi head; predictions L-1–L-5 (labelled X26a in the plan) | §5.1, §5.3; App. G; App. K |
| `X26c_fr3_lambda.py` | L2 weight-decay λ*(N) on FR3: the capacity anchor (X26c; L-4) | §5.1, §5.3; App. G; App. K |
| `X32_kernel.py` | Local (RBF kernel) psi head vs the global linear head (X32; X32-1–X32-5) | §5.1, §5.3; App. G; App. K |
| `X33_fr3_twoslot.py` | Two-slot psi head (global trend + local kernel), joint vs sequential (X33; X33-1–X33-4) | §5.1, §5.3; App. G; App. K |
| `X34_fr3_kernel.py` | Local kernel psi head on the main FR3 result (X34; P-FRK) | §5.1; App. G.4; App. K |
| `X35_recount_fr3.py` | Recount the FR3 relaxed-psi rows under the final two-slot head (X35; R-1); feeds Figure 3(a) | §5.1 (Table 3); Figure 3(a); App. G |
| `X36_recount_socket.py` | Recount the socket/ABB correction rows under the final heads (X36; R-2–R-4) | §5.3, §5.4; Tables 4-5; App. F.3, G |
| `X39_classical_socket.py` | Independent classical per-joint screw socket reference (LM) | §5.3; Table 4; App. F.3 |
| `X40_classical_socket.py` | Standalone classical PoE socket estimator under the matched Adam optimizer | §5.3; Table 4; App. F.3 |
| `X41_classical_ref.py` | Classical ABB range-only reference under the matched Adam optimizer; **external data** | §5.4; Table 5; App. F.3 |
| `X43_abb_lm.py` | Our ABB model fitted with the classical LM optimizer on the GN residual; **external data, diagnostic only** | §5.4; Table 5; App. F.3 |

*`run_main_experiments.py` was formerly named `run_all_experiments.py`
(v1.0–v1.1); renamed in v1.2 because the paper grew beyond that single file.*

**Revision round (`X22`–`X36`, capacity and locality of the law correction).** The
experiments that localize the seed instability of the learned law correction and fix
its parameterization ship here as well: `X22` (the strong learned baseline requested
in review); `X24`–`X26` (two mechanism hypotheses — the exponential winding and
unbounded magnitude — both refuted and reported); `X26b`/`X26c` (the capacity remedy:
an L2 weight-decay window and its dependence on training size); and `X32`–`X34` (the
locality remedy: a local kernel head, ending in the two-slot ψ head the appendix
recommends). The FR3 scripts auto-download `calib_data.pickle`; the socket-family
scripts (`X25`/`X25b`/`X26`/`X26b`/`X32`) read the MUKCa data under
`../newdata/kinematics_calibration/data/` (see **Data**). Script names follow the
plan's labels; the plan labelled the λ-sweep `X26a` while the file is
`X26b_lambda.py`. Registered as `x22`, `x24`, `x25`, `x25b`, `x26`, `x26a`, `x26a2`,
`x26c`, `x32`, `x32_5`, `x33`, `x34` in `repro/canonical.json`, and cited
row-by-row in [`predictions.md`](predictions.md).

`assets/lawkan_architecture.svg` is the architecture diagram shown at the top
(illustrative only; no result number depends on it).

`tools/figures/*.py` regenerate the paper's vector figures from the JSON dumps in this repo
(`repro/json/`, resolved through `repro/canonical.json`); they write the PDFs into the sibling paper
repo's `figures/` when it is present on disk. See `tools/figures/README.md`.

| Figure script | Paper item |
|---|---|
| `fig00_readme_summary.py` | README summary figure |
| `fig02_crossgroup.py` | Figure A1 cross-group characterization |
| `fig04_matrix_pareto.py` | Figure 2 opponent matrix |
| `fig05_sample_efficiency.py` | Figure A3 sample efficiency |
| `fig08_imu.py` | Figure A5 BROAD IMU |
| `fig09_node_capacity.py` | Figure A2 node-capacity curves |
| `fig10_x10_platforms.py` | Figure A6 three-platform FK |
| `fig12_noise_extrap.py` | Figure A4 noise/pole extrapolation |
| `fig13_learned_law.py` | Figure 3 learned law |
| `fig14_law_diagnostics.py` | Figure A9 law controls |
| `fig15_new_platforms.py` | Figure A8 ABB/MUKCa platforms |
| `fig16_kinova_seeds.py` | Figure A7 Kinova seed stability |
| `fig17_convergence.py` | Figure 4 convergence |
| `figstyle.py` | shared style/palette (helper) |
| `build_llms_full.py` | concatenate README/docs into `llms-full.txt` |
| `build_predictions_md.py` | regenerate `docs/predictions.md` from the paper supplement |
| `span_check.py` | span-criterion pre-training check used in §5.7 |
| `check_abb_data.py` | verify the external ABB xlsx name/size/SHA256 |

## Requirements

- Python 3.8+ (all scripts smoke-tested on Python 3.8.10)
- PyTorch (CUDA build recommended; CPU works but is slow)
- NumPy; pandas (real-data probes)
- SciPy (paired t-tests and statistics: without it t-statistics are still
  computed, but p-values print as `nan`, and the p-values are paper numbers)
- h5py (only for X5/X6/X8; BROAD HDF5 files)
- Figures: matplotlib and SciencePlots (`pip install matplotlib scienceplots`)
- X7, the VQF reference and the MUKCa/X10 probes: `roboticstoolbox-python`,
  `robotkinecal` and `vqf` (`pip install roboticstoolbox-python robotkinecal vqf`),
  plus the **pinned** `rtb-data==1.1.0` (rtb-data 2.0.0 breaks
  `rtb.models.URDF.Panda()`).

## Data

No data downloads for the synthetic core (§4, E16, E19): the generators
are built into the scripts. Two real datasets, both public:

- **BROAD** (X5/X6/X8; CC-BY 4.0, not redistributed):
  `git clone --depth 1 https://github.com/dlaidig/broad.git && cd broad &&
   git lfs install && git lfs pull --include="data_hdf5/*"`, then pass
  `--data_dir broad/data_hdf5`.
- **Robot calibration** (X7): `calib_data.pickle` (PhilNad/
  robot-arm-kinematic-calibration, MIT) is downloaded automatically by the
  script on first run.

## Running individual blocks

```bash
# synthetic core
python experiments/synthetic/run_main_experiments.py  | tee run_main.log   # ~35-45 min GPU, ~2-3 h CPU
python experiments/synthetic/run_extensions.py        | tee run_ext.log    # ~30-40 min GPU
python experiments/synthetic/E16_gl2_composition.py   | tee run_gl2.log    # ~4-6 min GPU
python experiments/synthetic/E19_sparse_edge.py       | tee run_e19.log    # full battery; --quick = CPU smoke
# real data
python experiments/real/X5_broad_real_imu.py   --data_dir broad/data_hdf5
python experiments/real/X6_broad_calibration.py --data_dir broad/data_hdf5 [--only c-1s]
python experiments/real/X7_robot_calibration.py                        # ~25-35 min CPU (smoke: --seeds 1 --sizes 80)
python experiments/real/X8_broad_seq.py        --data_dir broad/data_hdf5
```

Prefer the harness — it writes full logs, extracts the JSON, and snapshots the
environment (git revision, torch/CUDA, GPU, date) next to the results:

```bash
bash repro/run_all.sh stage1   # paper core: run_main, run_extensions, E16, E19, bayes
bash repro/run_all.sh stage2   # real data: X5, X6, X8, X9, X7
bash repro/run_all.sh stage3   # expansion (registered seeds): E20–E24
bash repro/run_all.sh stage4   # state-conditioned edges: E25, E25b
bash repro/run_all.sh stage5   # variable-angle theorem + full-SO(3): E29, E30
bash repro/run_all.sh stage6   # three-platform FK calibration: X10 (P48)
bash repro/run_all.sh stage7   # phase-2 pre-registered runs: E31 (P49), E32 (P50), E33 (floor metric)
bash repro/run_all.sh stage8   # phase-2 cont.: E31b/E31c/E31d, E33b/E33c, E34 (X7 recipe; journal-only)
bash repro/run_all.sh stage9   # MUKCa socket family: KUKA/Panda/Kinova (10 seeds), FR3 unit, cross-view
bash repro/run_all.sh stage10  # ABB RobotCali range-only (10 seeds)
bash repro/run_all.sh stage11  # X6b (TinyGC-style), X7 vqf reference, X16 isolated
bash repro/run_all.sh stage12  # appendix diagnostics: G2a/G2bc/G3/G4, X17, S8
bash repro/run_all.sh stage13  # validity package: X18/X18b, X19, X20/X21, POOL
```

Blocks are independent: to reproduce a single table, comment out the other
blocks and keep the shared preamble (geometry, data, models, protocol).

## What each block reproduces

**`run_main_experiments.py`**

| Block | Paper | Contents |
|---|---|---|
| E1 | §4.2.1 (Table 2) | Opponent matrix at n=2, 18 configurations (LawKAN vs MLP-Quat / MLP-6D / MLP-9D-SVD / plain MLP at several widths, incl. the RPMG row) + paired t-tests |
| E2 | App. A.5, B | Wall-clock per train step and per inference batch, parameter counts, FLOPs |
| E3.1 | App. E.3 | Chain length n=3 + paired t-test |
| E3.2 | App. E.3 | Input-noise curve |
| E3.3 | App. E.1 | Sample efficiency vs the strongest opponent |
| E3.4 | App. E.3 | Angular extrapolation (corrected sampler) |
| E4 | App. E.1 | Sample efficiency N = 50...1600, three models |
| E5 | App. E.3 | Addition in Z_n, n = 3...8 (clock-model / BKT check) |
| E6 | App. I.2 | Dead zones: q/||q|| normalization vs exp-map, 10 runs + min-norm trajectory trace |
| E7 | App. E.3 | Scope boundary: null-task and random-angle controls |
| E8 | App. E.3 | Error scaling along the composition chain |
| E9 | §4.2.1 | Opponent learning-rate sweep (registered prediction P18) |
| E10 | §4.2.1; App. D.4 | Monte-Carlo Bayes floor for the main task |
| E11 | §4.2.1; App. D.4 | Lie baseline: MLP -> so(3) -> exp readout, no group-law nodes (P19) |
| E12 | §4.2.1; App. D.4 | Per-seed values + paired t-tests for the key cells |
| E13 | §4.2.1; App. D.4 | Fourier-order ablation K = 1 vs 2 (P20) |
| E14 | App. E.3; App. D.4 | n=6 per-seed, 10 seeds, bimodality check (P22) |

**`run_extensions.py`**

| Block | Paper | Contents |
|---|---|---|
| X1 | §4.2.1 (Table 2) | RPMG opponent (Chen et al. 2022): manifold-aware projective gradient; independent rerun of the E1 row |
| X2 | App. D.4 | Non-Gaussian noise battery: uniform and Laplace noise at matched RMS |
| X3 | §4–§5; App. F, G | Applied probes: IMU dead-reckoning (E15, exact SU(2) edge, 27 params), 5-DOF FK calibration (per-joint SE(3) model), rigid-body feed-forward boundary (E17) + perturbation control |
| X4 | App. D.4 | Vector-Neurons opponent (Deng et al. 2021; minimal, simplified, labeled as such) |
| E18 | App. D.4 | Composition-operator ablation: BCH truncation ladder vs the exact Hamilton product + bit-equivalence of the flop-lean `rotate()` identity |

**`E19_sparse_edge.py`**: Hypothesis N, full battery: E19.0 analytic
exactness (float64), E19.1 main cell comp2 (P23), E19.2 K/feature ablation
(150p/54p/18p), E19.3 noise curve sigma = 0.01...0.20 (P24), E19.4 comp3
(P25), E19.5 chains n = 2,3,4, E19.6 sample efficiency N = 50...1600 (P26),
E19.7 weight-recovery diagnostic (P27), E19.8 cost (params, FLOPs under two
conventions, wall-clock). Standalone: does not import or modify the other
scripts.

**`X5_broad_real_imu.py`**: BROAD real-IMU probe: disturbed group by
default (undisturbed via `--groups undisturbed`), windows W = 20/100/286/1430
steps (~0.07/0.35/1.00/5.00 s at ~286 Hz), models BiasIntegrator /
RawIntegrator / exact SU(2) edge (`QuatIMU` in code; 27 params, 36 with the bias feature, ≥0.35 s windows, shared edge +
tree composition) /
the sparse 18-p edge (mask [1,5,8]) / flat MLPs, true SO(3) geodesic metric.

**`X6_broad_calibration.py`**: BROAD gyroscope calibration (undisturbed
group): raw windows, learn the fixed sensor defect omega_true = M(omega_raw
- b) by matching integrated orientations to opt_quat; no-learning
baselines RawIntegrator and BiasIntegrator; geodesic^2 loss and geodesic
metric (v1.1 fix; v1.0 rotvec-MSE numbers at the 1 s window are retracted).

**`X7_robot_calibration.py`**: real 7-DOF Franka calibration, 114 poses,
paired repeated splits (5 seeds) at train n = 20...80: Nominal / BaseFit /
PoE-LS (robotkinecal, the dataset authors' classical baseline) / MLP-residual
/ per-joint SE(3) model (base + per-joint quaternion corrections + tool, the same
identifiable parameter set as PoE-LS). Registered predictions P32–P34 are
printed as verdicts at the end.

**`X8_broad_seq.py`**: the two standard escapes from the flat-MLP long-
horizon failure on BROAD, under the identical X5 protocol: GRUIntegrator
(~4k params, recurrence must learn the group law) and ResidMLP (local
correction to the integrator output, TLIO/SINS-error philosophy).
Registered predictions P30/P31.

**`E16_gl2_composition.py`**: structure-transfer probe: does the LawKAN
design (Fourier phase edges -> Lie algebra -> exp-map -> multiplicative node)
transfer from SU(2) to GL+(2,R)? The compact rotation coordinate is
Fourier-native; the non-compact stretch/shear coordinates are not; the
K=1 vs K=2 comparison there is the falsifiable part.

The October 2026 blocks (X6b, X7 vqf, X10 P-D, MK1/Kinova/MK1c, RC1, POOL, X18–X21) follow
the same pattern; their paper locations are in the layout table above and in appendices F
(applied and validity probes) and G (learned-law controls). `repro/make_tables.py` prints a
mechanical table for every block, and `repro/check_canonical.py` verifies that each
canonical dump exists and is tracked.

## Revision and pole-fix branch (X22-X61)

The revision round (`X22-X36`) covers the learned baseline, the validation gate,
capacity/locality of the correction and the final two-slot recounts. The pole-fix branch
(`X37-X61`) covers the convergence grid, socket heads, chart-pole diagnosis, quaternion
features, regularization/EMA, the group-metric kernel and the T1/T2 confirmation. Both live
in `experiments/real/` and run on CPU in float64. The branch numbers are cited in
Sections 5.1-5.3 and the Conclusion. Per-block commands, tiers, devices and canonical dumps
are in `repro/registry.json`; `python repro/run_registry.py --tier full --only x46,x59`
runs a subset. The first-version embedded-component probes (E27/E28) are not part of this
artifact.

Key branch scripts (all self-dumping, CPU float64): `X37_broad_edge_capacity.py`,
`X38_perjoint_socket.py`, `X38b_kernel_socket.py`, `X38c_screw_heads.py`,
`X42_optimizer_matched.py`, `X44_fk_matched.py`, `X45_gyro_matched.py`,
`X46_convergence.py`, `X47_socket_joint_vs_seq.py`, `X52_reg_sweep.py`,
`X53_socket_convergence.py`, `X54_pole_check.py`, `X55_qfeat_fix.py`, `X56_kgrid.py`,
`X57_cells.py`, `X58_swa.py`, `X59_ema.py`, `X60_geo.py`, `X61_confirm.py`, `X61_all.py`.

## Sanity numbers

On the reference GPU the frozen runs produced (your numbers may wobble in the
last digit across CUDA/cuDNN versions; orderings and effect sizes should hold):

- **E1:** LawKAN **2.78 +/- 0.02 deg** vs MLP-Quat h=32 **3.36 +/- 0.06 deg**
  (paired t = 18.63, df = 4, p = 4.9e-05); MLP-Quat + RPMG **3.42 +/- 0.15 deg** (B1-fixed),
  indistinguishable from the plain head (p = 0.18), loses to LawKAN (p = 0.002)
- **E2:** LawKAN train step ~4.6-6.3 ms/it vs ~2.4-4.5 ms/it for the opponents
  (a ~1.4-1.9x wall-clock tax at 12x fewer FLOPs/example; environment-dependent)
- **E40 (matched):** Bayes floor 2.733 deg; LawKAN sits 0.05 deg above it (historical E10 floor: 2.697/0.09)
- **E10b / P16b:** hidden-angle Bayes floor **37.7 deg**: LawKAN 38.21 (+0.5),
  top opponent 46.48 (+8.8): near-optimal even when the angle is not an input
- **E11:** Lie baseline **3.29 +/- 0.10 deg** (10 seeds; 3.28 +/- 0.07 on 5 seeds): group-law nodes, not the exp-map
  alone, carry the result
- **E13:** K=2 is significantly *worse* than K=1 (**2.87 +/- 0.03 deg** on 10 seeds; 2.88 +/- 0.04 on 5 seeds, p = 6.3e-03)
- **E14:** at n=6 only 5/10 seeds converge (bimodal failure, reported as such)
- **E15 / §5.5.1 IMU (synthetic):** exact SU(2) edge **0.082 +/- 0.000 deg** (27 params)
  vs MLP h=32 **9.15 +/- 0.11 deg** (t = 117.5, p = 7.2e-05); at N=50 train
  samples: 0.089 deg vs 42.6 deg
- **FK / App. G.1 FK calibration (synthetic):** per-joint SE(3) model **0.159 +/- 0.095 deg**
  (150 params) vs MLP **0.661 +/- 0.267 deg**
- **E25 / E25b / App. G.2 rigid-body (ODE):** with only omega_0 a feed-forward edge
  stalls (LawKAN K=1 12.66 +/- 5.34 deg, seed-bimodal; FF-quad 1.58, FF-cubic
  0.84 deg); a **state-conditioned** quadratic edge (39 params) reaches the
  metric floor **0.0818 +/- 0.0001 deg** (= analytic integrator 0.0816), while a
  linear state edge stalls at 7.13 deg. The task is an ODE solution map, not a
  composition of known increments; control: a 1e-3 perturbation of omega_0
  diverges by only 0.097 deg (the task is not chaotic)
- **X4:** VN-MLP 12.12 +/- 1.28 deg vs LawKAN 2.79 deg (p ~ 1e-4)
- **E19.1 (P23):** sparse 18p **2.747 +/- 0.017 deg** vs dense 54p
  **2.790 +/- 0.018 deg** (paired t = -9.68, p = 6.4e-4, 5/5 seeds); distance
  to the matched Bayes floor +0.014 deg vs +0.057 deg (dense). Monotonic in parameters:
  150p -> 2.87 deg, 54p -> 2.79 deg, 18p -> 2.75 deg.
- **X5 / §5.5.1 (BROAD, real):** sparse 18-p edge **1.057 +/- 0.0002** (1 s) /
  **1.189 +/- 0.008** (5 s), deterministic, vs the dense edge's bimodal long
  horizon (5 s mean 25.1, per-seed 1.14/1.19/73.1; which seed fails is
  environment-dependent); flat MLP h=8/h=32: 0.534/0.552 (0.07 s), 47.2/58.8
  (1 s), 88.2/102.2 (5 s); training-free strapdown 0.394/1.148/1.191
  (0.07/1/5 s): flat one-shot regression has no way to buy the composition
  structure.
- **X6 / §5.5.2 (BROAD, undisturbed):** learned (M, b) 12p **0.588 / 1.100 /
  1.371 deg** at 0.07/0.35/1.00 s vs raw integration 0.594/1.161/1.453 and
  per-trial bias-only 0.593/1.144/1.374; it matches the classical LM fit
  (<=0.009 deg at every horizon) and gains **0.044 deg** over bias-only at
  0.35 s, from physically sane parameters, reproducible across seeds.
- **X7 / §5.1 (real Franka Research 3; script-specific protocol, not Table 3):** per-joint SE(3) model **1.61 / 1.57 / 1.43 / 1.45 mm** at
  train n = 20/40/60/80 vs PoE-LS 1.72/1.58/1.47/1.52 mm and BaseFit ~4.4 mm;
  nominal zero-shot 10.65 +/- 1.90 mm; P32 (vs MLP-residual) passes at n=20/40,
  statistically tied at n=60/80. The paper's Table 3 exact-node rows use the `x7bench`
  protocol: 1.53/1.49/1.37/1.40 mm at N=20/40/60/80.
- **X8 (BROAD):** GRU (~4k params, ~200x) 14.9 deg at 1 s / 78.7 deg at 5 s,
  fails at flat-MLP level; ResidMLP 1.55/2.58 deg; sparse 18-p edge 1.06/1.19 deg
  at 18 parameters and one parallel pass.
- **October 2026 blocks:** `X6b` matches the classical LM fit to ≤0.002° (deterministic agreement; registered as a
  TOST check); `X10` P-D rerun uses 10 seeds; Table 21 uses the 5-seed P48 run (`x10_p48`); `MK1`/`RC1`/`POOL` give Tables 4-5
  (isolated correction **1.14/1.12 mm** vs GN **1.29/1.33 mm** at N=60/80; pooled p<1e-10). `X18`–`X21`
  are the coning/tau/friction validity probes; the boundary cases are the Kinova socket
  (seed-unstable) and the inconclusive X21 friction proxy. The paper repo's `tools/check_numbers.py` re-verifies the headline
  numbers against `repro/canonical.json`.
