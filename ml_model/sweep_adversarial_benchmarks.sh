#!/bin/bash
#
# Sweep adversarial-benchmark training experiments using the combined
# SPEC + adversarial dataset (ronamol_spec_v3_adversarial_v1.csv).
#
# Runs (from ml_model/):
#   1) adversarial_only — train/test split within one adversarial benchmark
#   2) spec_holdout_adversarial — SPEC train, hold out one adversarial; vary OOD onboarding size
#   3) conflict — SPEC + a second adversarial program in training; test on a different adversarial
#
# Usage: from this directory,
#   ./sweep_adversarial_benchmarks.sh
# or:
#   bash sweep_adversarial_benchmarks.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_PATH="training_data/ronamol_spec_v3_adversarial_v1.csv"
PYTHON_SCRIPT="train_adversarial_benchmarks.py"

# All eight C adversarial workloads (names match the `benchmark` column in the CSV)
ADVERSARIAL_BENCHMARKS=(
  adversarial_branches
  icache_blast
  many_pages_streaming
  pow2_stride_benign
  pow2_stride_thrash
  ptrchase_rand
  serial_mul_chain
  stlf_misalign
)

# Onboarding: how many labeled points from the held-out benchmark are added to SPEC training
ONBOARDING_SIZES=(0 32 256 1024 2048 4096 8192)

# README-inspired pairs (interfering program in train vs held-out test benchmark)
# Format: conflict_bench,test_bench
CONFLICT_PAIRS=(
  "pow2_stride_benign,pow2_stride_thrash"
  "pow2_stride_thrash,pow2_stride_benign"
  "many_pages_streaming,ptrchase_rand"
  "ptrchase_rand,many_pages_streaming"
)

SWEEP_DIR="sweep_results/adversarial_spec_v3_combined"
mkdir -p "$SWEEP_DIR"

SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
{
  echo "Sweep started at $(date)"
  echo "Dataset: $DATASET_PATH"
  echo "Working directory: $SCRIPT_DIR"
} > "$SWEEP_LOG"

run_train() {
  local run_dir="$1"
  shift
  mkdir -p "$run_dir"
  echo "" | tee -a "$SWEEP_LOG"
  echo "RUN -> $run_dir" | tee -a "$SWEEP_LOG"
  echo "CMD: python $PYTHON_SCRIPT $*" | tee -a "$SWEEP_LOG"
  if python "$PYTHON_SCRIPT" "$@" -o "$run_dir" >"$run_dir/training_output.txt" 2>&1; then
    echo "OK  $run_dir" | tee -a "$SWEEP_LOG"
  else
    echo "FAIL $run_dir" | tee -a "$SWEEP_LOG"
  fi
}

echo "=========================================="
echo "Adversarial benchmark sweep"
echo "Dataset: $DATASET_PATH"
echo "Results: $SWEEP_DIR"
echo "=========================================="

# --- 1) Fit each adversarial benchmark using only its own rows ---
for b in "${ADVERSARIAL_BENCHMARKS[@]}"; do
  run_train \
    "$SWEEP_DIR/experiment_adversarial_only/ood_${b}" \
    -d "$DATASET_PATH" \
    --experiment adversarial_only \
    --ood-benchmark "$b"
done

# --- 2) SPEC holdout + onboarding curve per adversarial benchmark ---
for b in "${ADVERSARIAL_BENCHMARKS[@]}"; do
  for n in "${ONBOARDING_SIZES[@]}"; do
    run_train \
      "$SWEEP_DIR/experiment_spec_holdout_onboarding/ood_${b}/ood_train_size_${n}" \
      -d "$DATASET_PATH" \
      --experiment spec_holdout_adversarial \
      --train-benchmarks spec \
      --ood-benchmark "$b" \
      --ood-train-size "$n"
  done
done

# --- 3) Train on SPEC + one other adversarial; test on a different adversarial (conflict) ---
for pair in "${CONFLICT_PAIRS[@]}"; do
  IFS=',' read -r conflict_b test_b <<<"$pair"
  run_train \
    "$SWEEP_DIR/experiment_conflict/train_spec_plus_${conflict_b}/test_${test_b}" \
    -d "$DATASET_PATH" \
    --experiment spec_holdout_adversarial \
    --train-benchmarks "spec,${conflict_b}" \
    --ood-benchmark "$test_b" \
    --ood-train-size 0
done

# --- 4) Optional: all other adversarial programs + SPEC in train (no onboarding) ---
for b in "${ADVERSARIAL_BENCHMARKS[@]}"; do
  run_train \
    "$SWEEP_DIR/experiment_spec_plus_all_other_adversarial/test_${b}" \
    -d "$DATASET_PATH" \
    --experiment spec_holdout_adversarial \
    --train-benchmarks "spec,adversarial_except_ood" \
    --ood-benchmark "$b" \
    --ood-train-size 0
done

echo "Sweep finished at $(date)" | tee -a "$SWEEP_LOG"

SUMMARY_FILE="$SWEEP_DIR/summary.txt"
{
  echo "Adversarial benchmark sweep summary"
  echo "Generated at: $(date)"
  echo "Dataset: $DATASET_PATH"
  echo ""
} > "$SUMMARY_FILE"

while IFS= read -r -d '' metrics; do
  run_dir="$(dirname "$metrics")"
  echo "Run: $run_dir" >>"$SUMMARY_FILE"
  if command -v jq >/dev/null 2>&1; then
    jq -r '
      "  experiment: " + (.run_config.experiment // "n/a") +
      "\n  ood_benchmark: " + (.run_config.ood_benchmark // "n/a") +
      "\n  ood_train_size: " + ((.run_config.ood_train_size // 0) | tostring) +
      "\n  best_eval_loss: " + ((.run_config.best_eval_loss // "n/a") | tostring) +
      "\n  best_percent_error: " + ((.run_config.best_percent_error // "n/a") | tostring)
    ' "$metrics" 2>/dev/null >>"$SUMMARY_FILE" || echo "  (jq parse failed)" >>"$SUMMARY_FILE"
  else
    echo "  (install jq for structured metrics summary; showing last Percent Error line)" >>"$SUMMARY_FILE"
    if [[ -f "$run_dir/training_output.txt" ]]; then
      tail -30 "$run_dir/training_output.txt" | grep "Percent Error" | tail -1 >>"$SUMMARY_FILE" || true
    fi
  fi
  echo "" >>"$SUMMARY_FILE"
done < <(find "$SWEEP_DIR" -name metrics.json -print0 | sort -z)

echo "Summary written to $SUMMARY_FILE"
