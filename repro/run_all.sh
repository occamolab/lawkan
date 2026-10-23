#!/usr/bin/env bash
# Reproducibility harness: runs the experiment scripts, writes full stdout logs,
# extracts the machine-readable SUMMARY-JSON of each, and records the environment
# (git revision, torch/CUDA, GPU, date).
#
#   bash repro/run_all.sh stage1   # paper core: run_main, run_extensions, E16, E19
#   bash repro/run_all.sh stage2   # real data: X5, X6, X8, X7
#   bash repro/run_all.sh stage3   # expansion (registered seeds): E20--E24
#   bash repro/run_all.sh stage4   # state-conditioned edges: E25, E25b
#   bash repro/run_all.sh stage5   # variable-angle theorem + full-SO(3): E29, E30
#   bash repro/run_all.sh stage6   # three-platform FK calibration (X10, P48)
#   bash repro/run_all.sh stage7   # phase-2: E31 (P49), E32 (P50), E33 (floor metric)
#   bash repro/run_all.sh stage8   # phase-2 cont.: E31b/E31c/E31d, E33b/E33c, E34 (X7 recipe)
#   bash repro/run_all.sh stage9   # MUKCa socket family (KUKA/Panda/Kinova 10 seeds, FR3 unit, cross-view)
#   bash repro/run_all.sh stage10  # ABB RobotCali range-only (10 seeds)
#   bash repro/run_all.sh stage11  # X6b (TinyGC-style), X7 vqf reference, X16 isolated
#   bash repro/run_all.sh stage12  # appendix diagnostics: G2a/G2bc/G3/G4, X17, S8
#   bash repro/run_all.sh stage13  # validity package: X18/X18b (coning), X19 (tau-psi), X20/X21 (friction), POOL
#   bash repro/run_all.sh stage14  # revision round: X22-X34 (capacity and locality of the correction)
#   bash repro/run_all.sh stage15  # pole-fix branch: X35-X61 (registry-driven)
#   bash repro/run_all.sh headline # main-text tables/figures end to end (~overnight)
#   bash repro/run_all.sh headline-smoke  # minutes-scale sanity of the headline blocks
#
# Outputs (under repro/):
#   logs/<script>.<ts>.log        full stdout/stderr
#   json/<script>.<ts>.json       extracted SUMMARY-JSON
#   env.<ts>.txt                  environment snapshot
#   MANIFEST.<ts>.tsv             script, command, exit, wall_s, json path
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
BROAD="${BROAD:-$ROOT/../broad/data_hdf5}"
CALIB="${CALIB:-$ROOT/data/calib_data.pickle}"
STAGE="${1:-stage1}"
ONLY="${2:-}"
TS="$(date +%Y%m%d-%H%M%S)"
R="$ROOT/repro"; L="$R/logs"; J="$R/json"
mkdir -p "$L" "$J"
cd "$ROOT"

if pgrep -f 'ollama serve' >/dev/null 2>&1; then
  echo "WARNING: ollama serve is running and may compete for GPU."
fi

{
  echo "date: $(date -Is)"
  echo "stage: $STAGE"
  echo "pwd: $ROOT"
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null) $(git describe --always --dirty 2>/dev/null)"
  echo "--- git status ---"; git status --porcelain 2>/dev/null
  echo "python: $($PY --version 2>&1)"
  "$PY" - <<'EOF' 2>&1
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
EOF
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$L/env.$TS.txt" 2>&1

MAN="$R/MANIFEST.$TS.tsv"
printf 'script\tcmd\texit\twall_s\tjson\n' > "$MAN"
FAILED=0

run_one() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then return 0; fi
  local log="$L/$name.$TS.log" jf="$J/$name.$TS.json"
  echo "[$STAGE] === $name: $* ==="
  local t0 t1 rc ex
  t0=$(date +%s)
  "$@" >"$log" 2>&1; rc=$?
  t1=$(date +%s)
  if [ "${NO_EXTRACT:-0}" = "1" ]; then
    jf="self-dump:$name"
  elif "$PY" "$R/extract_json.py" "$log" "$jf" >/dev/null 2>&1; then
    ex=0
  else
    ex=$?; jf=""
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$*" "$rc" "$((t1 - t0))" "$jf" >> "$MAN"
  [ "$rc" -ne 0 ] && FAILED=1
  echo "[$STAGE] $name: rc=$rc wall=$((t1 - t0))s json=${jf:-NONE(extract rc=$ex)}"
}

case "$STAGE" in
  stage1)
    run_one run_main "$PY" experiments/synthetic/run_main_experiments.py
    run_one run_ext  "$PY" experiments/synthetic/run_extensions.py
    run_one e16      "$PY" experiments/synthetic/E16_gl2_composition.py
    run_one e19      "$PY" experiments/synthetic/E19_sparse_edge.py
    run_one bayes    "$PY" repro/bayes_p16b.py
    ;;
  stage2)
    run_one x5 "$PY" experiments/real/X5_broad_real_imu.py --data_dir "$BROAD"
    run_one x6 "$PY" experiments/real/X6_broad_calibration.py --data_dir "$BROAD"
    run_one x8 "$PY" experiments/real/X8_broad_seq.py --data_dir "$BROAD"
    run_one x8t "$PY" experiments/real/X8_broad_seq.py --data_dir "$BROAD" --gru_tuned
    run_one x9 "$PY" experiments/real/X9_imu_tuning_sensitivity.py --data_dir "$BROAD"
    run_one x7 "$PY" experiments/real/X7_robot_calibration.py --data "$CALIB"
    ;;
  stage3)
    # expansion experiments at their REGISTERED seeds (see the experiment journal):
    run_one e20 "$PY" experiments/synthetic/E20_higher_harmonic.py --seed_start 10 --seeds 5
    run_one e21 "$PY" experiments/synthetic/E21_gl2_support_discovery.py --seed_start 10 --seeds 5
    run_one e22 "$PY" experiments/synthetic/E22_se3_composition.py --seed_start 10 --seeds 5
    run_one e23 "$PY" experiments/synthetic/E23_se3_feature_ablation.py --seed_start 20 --seeds 5
    run_one e24 "$PY" experiments/synthetic/E24_order_k_scaling.py --seed_start 20 --seeds 5
    ;;
  stage4)
    # state-conditioned edge (E25) + feed-forward control (E25b)
    run_one e25   "$PY" experiments/synthetic/E25_state_rigid.py
    run_one e25b  "$PY" experiments/synthetic/E25b_ff_poly_control.py
    ;;
  stage5)
    # variable-angle theorem (E29), full-SO(3) readout (E30)
    run_one e29  "$PY" experiments/synthetic/E29_feature_span.py
    run_one e30  "$PY" experiments/synthetic/E30_full_so3.py
    ;;
  stage6)
    # three-platform FK calibration (X10), pre-registered replication P48 (fresh seeds 50-54)
    run_one x10 "$PY" experiments/real/X10_fk_calib_real.py --seed_start 50 --seeds 5
    ;;
  stage7)
    # phase-2 (pre-registered P49/P50/E33): scripts self-dump via repro/dump.py
    NO_EXTRACT=1 run_one e31 "$PY" experiments/synthetic/E31_node_isolation.py
    NO_EXTRACT=1 run_one e32 "$PY" experiments/synthetic/E32_opponent_tuning.py
    NO_EXTRACT=1 run_one e33 "$PY" experiments/synthetic/E33_floor_reported_metric.py
    ;;
  stage8)
    # phase-2 cont. (pre-registered): E31b/E31c/E31d, E33b/E33c, E34 (X7 recipe)
    NO_EXTRACT=1 run_one e31b "$PY" experiments/synthetic/E31b_node_mlp_h4.py
    NO_EXTRACT=1 run_one e31c "$PY" experiments/synthetic/E31c_sparse_edge.py
    NO_EXTRACT=1 run_one e31d "$PY" experiments/synthetic/E31d_pruning_curve.py
    NO_EXTRACT=1 run_one e33b "$PY" experiments/synthetic/E33b_metric_loss.py
    NO_EXTRACT=1 run_one e33c "$PY" experiments/synthetic/E33c_metric_loss_stable.py
    NO_EXTRACT=1 run_one e34 "$PY" experiments/real/E34_x7_recipe.py --prune
    ;;
  stage9)
    # MUKCa socket-consistency family (P-MK7/P-MK8/P-D): front trio at 10 seeds,
    # FR3 unit at 5, KUKA cross-view at 5 (within-robot replication)
    NO_EXTRACT=1 run_one mk1      "$PY" experiments/real/MK1_mukca.py --platforms kuka14,panda_1,kinova --seeds 10
    NO_EXTRACT=1 run_one mk1fr3   "$PY" experiments/real/MK1_mukca.py --platforms fr3 --seeds 5
    NO_EXTRACT=1 run_one mk1views "$PY" experiments/real/MK1_mukca.py --platforms kuka14_left_back,kuka14_raised_left --seeds 5
    ;;
  stage10)
    # ABB RobotCali draw-wire range-only (P-D, 10 seeds)
    NO_EXTRACT=1 run_one rc1 "$PY" experiments/real/RC1_robotcali.py --seeds 10
    ;;
  stage11)
    # calibration competitor, AHRS reference and isolated-law diagnostics
    NO_EXTRACT=1 run_one x6b    "$PY" experiments/real/X6b_tinygc_style.py --data_dir "$BROAD"
    NO_EXTRACT=1 run_one x7vqf  "$PY" experiments/real/X7_vqf_reference.py --data_dir "$BROAD" --groups disturbed --blocks a-short,b2-035s,b-1s,c-5s,d-short,d-035s
    NO_EXTRACT=1 run_one x7vqfu "$PY" experiments/real/X7_vqf_reference.py --data_dir "$BROAD" --groups undisturbed --blocks a-short,b2-035s,b-1s,c-5s
    NO_EXTRACT=1 run_one x16iso "$PY" experiments/real/X16_psi_diagnostics.py --isolated
    ;;
  headline)
    # main-text tables and figures end to end (final settings; CPU reference)
    "$PY" repro/run_registry.py --tier headline --mode reference
    ;;
  stage12)
    # appendix diagnostics (G-series, sparse law, UR10 relaxed boundary); self-dumping scripts
    NO_EXTRACT=1 run_one g2a   "$PY" experiments/real/G2a_ur10_isolated.py
    NO_EXTRACT=1 run_one g2bc  "$PY" experiments/real/G2bc_isolated_psi.py
    NO_EXTRACT=1 run_one g3    "$PY" experiments/real/G3_orthogonality.py
    NO_EXTRACT=1 run_one g4    "$PY" experiments/real/G4_gauge_compare.py
    NO_EXTRACT=1 run_one x17   "$PY" experiments/real/X17_sparse_law.py
    NO_EXTRACT=1 run_one s8    "$PY" experiments/real/S8_ur10_relaxed.py
    ;;
  stage13)
    # validity package: coning (X18/X18b), tau-psi verdict (X19_recover reads the
    # archived training log + canonical g2bc_fig dump), friction (X20/X21), pooled verdict
    NO_EXTRACT=1 run_one x18    "$PY" experiments/real/X18_coning.py
    NO_EXTRACT=1 run_one x18b   "$PY" experiments/real/X18b_coning_constrained.py
    NO_EXTRACT=1 run_one x19rec "$PY" experiments/real/X19_recover.py
    NO_EXTRACT=1 run_one x20    "$PY" experiments/real/X20_fric_synth.py
    NO_EXTRACT=1 run_one x21    "$PY" experiments/real/X21_fric_proxy.py
    NO_EXTRACT=1 run_one pool   "$PY" experiments/real/POOL_verdict.py
    ;;
  stage15)
    # pole-fix / convergence branch, CPU float64, registry-driven
    "$PY" repro/run_registry.py --mode reference --only \
      x35,x36a,x36c,x36e,x36seq,x37,x38,x38b,x38c,x42,x43,x44,x44_ur5,x45,\
x46,x46a,x46b,x46b2,x47,x52,x53sock,x54,x55,x56,x57,x58,x59,x60,x61,x61b
    ;;
  stage14)
    # revision round: capacity and locality of the law correction (CPU, float64 by design)
    NO_EXTRACT=1 run_one x22   "$PY" experiments/real/X22_learned_baseline.py
    NO_EXTRACT=1 run_one x24   "$PY" experiments/real/X24_fr3_gate.py
    NO_EXTRACT=1 run_one x25   "$PY" experiments/real/X25_psi_bound.py
    NO_EXTRACT=1 run_one x25b  "$PY" experiments/real/X25b_weight_corr.py
    NO_EXTRACT=1 run_one x26   "$PY" experiments/real/X26_fullbound.py --seed-list 8,9
    NO_EXTRACT=1 run_one x26b  "$PY" experiments/real/X26b_lambda.py --l2-list 1e-3,3e-3,1e-2
    NO_EXTRACT=1 run_one x26b2 "$PY" experiments/real/X26b_lambda.py --l2-list 1e-4,3e-4
    NO_EXTRACT=1 run_one x26c  "$PY" experiments/real/X26c_fr3_lambda.py --l2-list 0,1e-4,3e-4,1e-3,3e-3,1e-2
    NO_EXTRACT=1 run_one x32   "$PY" experiments/real/X32_kernel.py --seed-list 0,1,2,8,9
    NO_EXTRACT=1 run_one x32_5 "$PY" experiments/real/X32_kernel.py --seed-list 0,8,9 --l2 1e-4
    NO_EXTRACT=1 run_one x33   "$PY" experiments/real/X33_fr3_twoslot.py
    NO_EXTRACT=1 run_one x34   "$PY" experiments/real/X34_fr3_kernel.py
    ;;
  headline-smoke)
    # minutes-scale sanity for the headline blocks that support --quick
    NO_EXTRACT=1 run_one hsmoke_mk1 "$PY" experiments/real/MK1_mukca.py --platforms kuka14 --quick
    NO_EXTRACT=1 run_one hsmoke_mk1fr3 "$PY" experiments/real/MK1_mukca.py --platforms fr3 --quick
    NO_EXTRACT=1 run_one hsmoke_mk1views "$PY" experiments/real/MK1_mukca.py --platforms kuka14_left_back,kuka14_raised_left --quick
    NO_EXTRACT=1 run_one hsmoke_rc1 "$PY" experiments/real/RC1_robotcali.py --quick
    NO_EXTRACT=1 run_one hsmoke_x6b "$PY" experiments/real/X6b_tinygc_style.py --data_dir "$BROAD" --quick
    NO_EXTRACT=1 run_one hsmoke_x7vqf "$PY" experiments/real/X7_vqf_reference.py --data_dir "$BROAD" --quick
    NO_EXTRACT=1 run_one hsmoke_x16 "$PY" experiments/real/X16_psi_diagnostics.py --isolated --quick
    NO_EXTRACT=1 run_one hsmoke_x10 "$PY" experiments/real/X10_fk_calib_real.py --quick
    ;;
  *)
    echo "usage: run_all.sh [stage1|...|stage15|headline|headline-smoke]"; exit 1 ;;
esac

echo "[$STAGE] ALL DONE. manifest=$MAN"
[ "$FAILED" -eq 0 ] || { echo "[$STAGE] FAILED blocks present (see $MAN)"; exit 1; }
