#!/bin/bash
#
# Sweep every combination of the ablation flags accepted by train_ablations.py:
#   --drop-cache-cdf      drop the 50/75/95-percentile cache fetch/exec latency columns
#   --drop-stale-features drop prog_frac_simd, prog_frac_branch, prog_frac_other
#
# With N=2 flags we enumerate 2^N = 4 runs. Each run produces checkpoint.pt,
# scaler.joblib, and metrics.json under its own subdirectory, named by the set
# of dropped groups.
#
# Usage (from ml_model/):
#   ./sweep_ablations.sh
# or:
#   bash sweep_ablations.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_PATH="training_data/ronamol_spec_training_data_v3.csv"
PYTHON_SCRIPT="train_ablations.py"

# Ordered list of ablation flags. Order determines the bitmask indexing below
# and the ordering of the label suffix for each run.
GROUP_NAMES=(cache_cdf_features stale_features)
GROUP_FLAGS=(
  "--drop-cache-cdf-features"
  "--drop-stale-features"
)
N_GROUPS=${#GROUP_NAMES[@]}
N_COMBOS=$(( 1 << N_GROUPS ))

SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="sweep_results/ablations_${SWEEP_TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
{
  echo "Ablation sweep started at $(date)"
  echo "Dataset:  $DATASET_PATH"
  echo "Script:   $PYTHON_SCRIPT"
  echo "Groups:   ${GROUP_NAMES[*]}"
  echo "Combos:   $N_COMBOS"
} > "$SWEEP_LOG"

echo "=========================================="
echo "Ablation sweep"
echo "Dataset: $DATASET_PATH"
echo "Results: $SWEEP_DIR"
echo "Groups:  ${GROUP_NAMES[*]}  ($N_COMBOS combinations)"
echo "=========================================="

run_one() {
  # run_one <run_dir> <label> <extra_arg_1> <extra_arg_2> ...
  local run_dir="$1"
  local label="$2"
  shift 2
  mkdir -p "$run_dir"

  echo ""
  echo "-> $label"
  echo "   dir: $run_dir"
  {
    echo ""
    echo "RUN -> $run_dir"
    echo "LABEL: $label"
    echo "CMD:  python $PYTHON_SCRIPT -d $DATASET_PATH -o $run_dir $*"
  } >> "$SWEEP_LOG"

  if python "$PYTHON_SCRIPT" \
      -d "$DATASET_PATH" \
      -o "$run_dir" \
      "$@" \
      > "$run_dir/training_output.txt" 2>&1; then
    echo "   OK"
    echo "OK  $run_dir" >> "$SWEEP_LOG"
  else
    echo "   FAIL (see $run_dir/training_output.txt)"
    echo "FAIL $run_dir" >> "$SWEEP_LOG"
  fi
}

# --- Enumerate the 2^N combinations of the bitmasked group flags ---
for ((mask = 0; mask < N_COMBOS; mask++)); do
  extra_args=()
  dropped=()
  for ((i = 0; i < N_GROUPS; i++)); do
    if (( (mask >> i) & 1 )); then
      extra_args+=("${GROUP_FLAGS[$i]}")
      dropped+=("${GROUP_NAMES[$i]}")
    fi
  done

  if (( ${#dropped[@]} == 0 )); then
    label="keep_all"
  else
    # Join dropped group names with '+' for a readable directory label.
    label="drop_$(IFS=+; echo "${dropped[*]}")"
  fi

  run_dir="$SWEEP_DIR/mask_$(printf "%02d" "$mask")_${label}"
  echo ""
  echo "[$((mask + 1))/$N_COMBOS] bitmask run"
  run_one "$run_dir" "$label" "${extra_args[@]}"
done

# --- Standalone --drop-all-features run (not a bitmask combination). This flag
# drops the union of program-feature and cache-feature columns, superseding the
# cache_cdf and stale feature groups, so it only makes sense to activate alone.
echo ""
echo "[extra] --drop-all-features standalone run"
ALL_RUN_DIR="$SWEEP_DIR/extra_drop_all_features"
run_one "$ALL_RUN_DIR" "drop_all_features" "--drop-all-features"

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Results saved in: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

# --- Build a CSV + text summary sorted by best eval percent error. ---
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
SUMMARY_TXT="$SWEEP_DIR/summary.txt"

{
  echo "mask,label,num_features,best_eval_loss,best_percent_error,drop_cache_cdf_features,drop_stale_features,drop_all_features"
} > "$SUMMARY_CSV"

python - "$SWEEP_DIR" "$SUMMARY_CSV" "$SUMMARY_TXT" <<'PY'
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
txt_path = Path(sys.argv[3])

rows = []
for run_dir in sorted(sweep_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    if not (run_dir.name.startswith("mask_") or run_dir.name.startswith("extra_")):
        continue
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        continue
    try:
        with metrics_path.open() as f:
            m = json.load(f)
    except Exception as e:
        print(f"  (skip {run_dir.name}: {e})")
        continue

    if run_dir.name.startswith("mask_"):
        parts = run_dir.name.split("_", 2)
        mask = parts[1] if len(parts) > 1 else ""
        label = parts[2] if len(parts) > 2 else run_dir.name
    else:
        mask = "extra"
        label = run_dir.name[len("extra_"):] or run_dir.name

    flags = m.get("ablation_flags", {})
    rows.append({
        "mask": mask,
        "label": label,
        "num_features": m.get("num_features"),
        "best_eval_loss": m.get("best_eval_loss"),
        "best_percent_error": m.get("best_percent_error"),
        "drop_cache_cdf_features": int(bool(flags.get("drop_cache_cdf_features"))),
        "drop_stale_features":     int(bool(flags.get("drop_stale_features"))),
        "drop_all_features":       int(bool(flags.get("drop_all_features"))),
    })

def _sort_key(r):
    v = r.get("best_percent_error")
    return (v is None, v if v is not None else float("inf"))

rows.sort(key=_sort_key)

with csv_path.open("a") as f:
    for r in rows:
        f.write(
            f"{r['mask']},{r['label']},"
            f"{r['num_features']},{r['best_eval_loss']},{r['best_percent_error']},"
            f"{r['drop_cache_cdf_features']},{r['drop_stale_features']},{r['drop_all_features']}\n"
        )

with txt_path.open("w") as f:
    f.write("Ablation sweep summary (sorted by best_percent_error ascending)\n")
    f.write(f"Results dir: {sweep_dir}\n")
    f.write(f"Runs:        {len(rows)}\n\n")
    header = f"{'mask':>4}  {'pct_err':>8}  {'eval_loss':>10}  {'nfeat':>5}  label\n"
    f.write(header)
    f.write("-" * len(header) + "\n")
    for r in rows:
        pe = r["best_percent_error"]
        el = r["best_eval_loss"]
        nf = r["num_features"]
        pe_s = f"{pe:.3f}" if isinstance(pe, (int, float)) else "n/a"
        el_s = f"{el:.5f}" if isinstance(el, (int, float)) else "n/a"
        nf_s = f"{nf}" if nf is not None else "n/a"
        f.write(f"{r['mask']:>4}  {pe_s:>8}  {el_s:>10}  {nf_s:>5}  {r['label']}\n")

print(f"Wrote CSV summary to {csv_path}")
print(f"Wrote text summary to {txt_path}")
PY

echo "Summary CSV: $SUMMARY_CSV"
echo "Summary TXT: $SUMMARY_TXT"
