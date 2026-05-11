#!/usr/bin/env bash
# run_sweep_parallel.sh
#
# For each row of sim_region_param_sweep.csv, pull the matching trace from S3,
# run the anamol analytical models in single-config mode, compute all training
# features, and append a row to training/training_data.csv.
#
# Parallelised with GNU parallel — one row per worker.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
ANAMOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEREGRINE_ROOT="$(dirname "$ANAMOL_ROOT")"

SWEEP_CSV="${SWEEP_CSV:-$PEREGRINE_ROOT/sim_region_param_sweep.csv}"
RESULTS_CSV="${RESULTS_CSV:-/home/ubuntu/peregrine-gem5/configs/peregrine/sweep_outputs_v3/sweep_results.csv}"
ANAMOL_BIN="${ANAMOL_BIN:-$ANAMOL_ROOT/anamol}"

TRAINING_DIR="${TRAINING_DIR:-$ANAMOL_ROOT/training}"
TRAINING_CSV="${TRAINING_CSV:-$TRAINING_DIR/training_data.csv}"
TRAINING_LOCK="${TRAINING_LOCK:-$TRAINING_DIR/.training_data.lock}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$ANAMOL_ROOT/.cache/sweep_scratch}"
INDEX_TSV="${INDEX_TSV:-$SCRATCH_ROOT/rows.tsv}"
PARALLEL_LOG="${PARALLEL_LOG:-$TRAINING_DIR/.parallel.joblog}"

JOBS="${JOBS:-$(nproc)}"
ROW_LIMIT="${ROW_LIMIT:-0}"   # 0 = all rows

# ── Pre-flight ────────────────────────────────────────────────────────────────
if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required (apt install parallel)" >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required (for trace download)" >&2
  exit 1
fi
if [[ ! -x "$ANAMOL_BIN" ]]; then
  echo "anamol binary missing or not executable: $ANAMOL_BIN" >&2
  echo "Build with: (cd $ANAMOL_ROOT && make anamol)" >&2
  exit 1
fi
if [[ ! -f "$SWEEP_CSV" ]]; then
  echo "Sweep CSV not found: $SWEEP_CSV" >&2
  exit 1
fi
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "Results (CPI) CSV not found: $RESULTS_CSV" >&2
  exit 1
fi

mkdir -p "$TRAINING_DIR" "$SCRATCH_ROOT"

# ── Build the row index (row_id \t row_csv \t cpi) once up front ──────────────
echo "[driver] Building row index → $INDEX_TSV" >&2
python3 "$ANAMOL_ROOT/build_cpi_index.py" \
  --sweep-csv "$SWEEP_CSV" \
  --results-csv "$RESULTS_CSV" \
  -o "$INDEX_TSV" \
  ${ROW_LIMIT:+--row-limit "$ROW_LIMIT"}

TOTAL_ROWS=$(wc -l < "$INDEX_TSV")
echo "[driver] $TOTAL_ROWS rows to process, $JOBS parallel workers" >&2

# ── Worker wrapper (invoked by GNU parallel) ─────────────────────────────────
# We can't easily pass an interpreter + arg list through `parallel` field
# splitting, so define a bash function.
export ANAMOL_ROOT PEREGRINE_ROOT TRAINING_CSV TRAINING_LOCK SCRATCH_ROOT

run_one() {
  local row_id="$1"
  local row_csv="$2"
  local cpi="$3"
  local scratch="$SCRATCH_ROOT/row_${row_id}"
  # PYTHONPATH lets process_sweep_row.py do `import registry / utils / models`.
  PYTHONPATH="$ANAMOL_ROOT/python:${PYTHONPATH:-}" \
    python3 "$ANAMOL_ROOT/python/process_sweep_row.py" \
      "$row_id" "$row_csv" "$cpi" \
      "$TRAINING_CSV" "$TRAINING_LOCK" "$scratch"
}
export -f run_one

# ── Launch ───────────────────────────────────────────────────────────────────
# --colsep '\t'     : split our TSV into {1}={row_id} {2}={row} {3}={cpi}
# --joblog ...      : resume-friendly record of completed rows
# --line-buffer     : interleaved but readable stdout/stderr
# --halt never      : keep going even if a worker exits non-zero (worker
#                     already swallows its own errors, but belt + suspenders)
parallel \
  --colsep '\t' \
  --jobs "$JOBS" \
  --joblog "$PARALLEL_LOG" \
  --line-buffer \
  --halt never \
  run_one {1} {2} {3} \
  :::: "$INDEX_TSV"

echo "[driver] Done. Output: $TRAINING_CSV" >&2
