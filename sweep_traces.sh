#!/usr/bin/env bash
# Generate cache latencies, branch predictor rates, and features for all traces in parallel.
#
# This script processes all trace files in TRACE_DIR to:
# 1. Generate cache latencies for all 100 configurations (5×5×4 combinations)
# 2. Generate branch predictor rates for 2 types (local, tage)
# 3. Generate ronamol features (program features + cache/BP summaries)
#
# Uses GNU parallel to run jobs concurrently.
#
# Environment variables (with defaults):
# - TRACE_DIR: Directory containing trace subdirs with trace.csv files
# - PEREGRINE_ROOT: Root directory containing gen_bp_rate.py and gen_cache_latency.py
#
# Each trace directory should contain a trace.csv file.
# Output files are created in each trace directory (see gen_cache_latency.py,
# gen_bp_rate.py, ronamol/python/gen_features.py):
#   - cache_latencies/l1i_<l1i_kb>_l1d_<l1d_kb>_l2_<l2_kb>_cache_latencies.npy
#   - bp_rates/<local|tage>_bp_rate.npy
#   - ronamol/program_features.csv
#   - ronamol/cache_latency_summary.csv
#   - ronamol/bp_rates_summary.csv

set -euo pipefail

if ! command -v parallel &>/dev/null; then
  echo "GNU parallel is required. Install with: apt install parallel" >&2
  exit 1
fi

# Default configurations
PEREGRINE_ROOT="${PEREGRINE_ROOT:-/home/ubuntu/peregrine}"
TRACE_DIR="${TRACE_DIR:-$PEREGRINE_ROOT/traces}"

# Cache configuration arrays (from gen_cache_latency.py)
L1_KB=(16 32 64 128 256)    # L1I and L1D sizes
L2_KB=(512 1024 2048 4096)  # L2 sizes

# Branch predictor types (from gen_bp_rate.py)
BP_TYPES=("local" "tage")

export PEREGRINE_ROOT TRACE_DIR L1_KB L2_KB BP_TYPES

if [[ ! -d "$PEREGRINE_ROOT" ]]; then
  echo "PEREGRINE_ROOT not found: $PEREGRINE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$TRACE_DIR" ]]; then
  echo "TRACE_DIR not found: $TRACE_DIR" >&2
  exit 1
fi

# Verify required Python scripts exist
if [[ ! -f "$PEREGRINE_ROOT/gen_cache_latency.py" ]]; then
  echo "gen_cache_latency.py not found in $PEREGRINE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PEREGRINE_ROOT/gen_bp_rate.py" ]]; then
  echo "gen_bp_rate.py not found in $PEREGRINE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PEREGRINE_ROOT/ronamol/python/gen_features.py" ]]; then
  echo "gen_features.py not found in $PEREGRINE_ROOT/ronamol/python/" >&2
  exit 1
fi

cd "$PEREGRINE_ROOT"

# Function to run cache latency simulation for one configuration
run_cache_latency() {
  local trace_file="$1"
  local l1i_size="$2"
  local l1d_size="$3"
  local l2_size="$4"

  if [[ ! -f "$trace_file" ]]; then
    echo "Trace file not found: $trace_file" >&2
    return 1
  fi

  local trace_dir
  trace_dir="$(dirname "$trace_file")"
  local output_dir="$trace_dir/cache_latencies"
  local output_file="$output_dir/l1i_${l1i_size}_l1d_${l1d_size}_l2_${l2_size}_cache_latencies.npy"

  # Skip if output already exists
  if [[ -f "$output_file" ]]; then
    echo "Cache latency output already exists, skipping: $output_file"
    return 0
  fi

  echo "Generating cache latencies: L1I=${l1i_size}KB, L1D=${l1d_size}KB, L2=${l2_size}KB for $(basename "$trace_dir")"
  
  python3 "$PEREGRINE_ROOT/gen_cache_latency.py" \
    --trace "$trace_file" \
    --l1i-size "$l1i_size" \
    --l1d-size "$l1d_size" \
    --l2-size "$l2_size"
}

# Function to run branch predictor simulation for one type
run_bp_rate() {
  local trace_file="$1"
  local bp_type="$2"

  if [[ ! -f "$trace_file" ]]; then
    echo "Trace file not found: $trace_file" >&2
    return 1
  fi

  local trace_dir
  trace_dir="$(dirname "$trace_file")"
  local output_dir="$trace_dir/bp_rates"
  local output_file="$output_dir/${bp_type}_bp_rate.npy"

  # Skip if output already exists
  if [[ -f "$output_file" ]]; then
    echo "Branch predictor output already exists, skipping: $output_file"
    return 0
  fi

  echo "Generating branch predictor rates: ${bp_type} for $(basename "$trace_dir")"
  
  python3 "$PEREGRINE_ROOT/gen_bp_rate.py" \
    --trace "$trace_file" \
    --branch-predictor "$bp_type"
}

# Function to run feature generation for one trace
run_features() {
  local trace_file="$1"

  if [[ ! -f "$trace_file" ]]; then
    echo "Trace file not found: $trace_file" >&2
    return 1
  fi

  local trace_dir
  trace_dir="$(dirname "$trace_file")"
  local output_dir="$trace_dir/ronamol"
  local program_features_file="$output_dir/program_features.csv"
  local cache_summary_file="$output_dir/cache_latency_summary.csv"
  local bp_summary_file="$output_dir/bp_rates_summary.csv"

  if [[ -f "$program_features_file" && -f "$cache_summary_file" && -f "$bp_summary_file" ]]; then
    echo "Feature outputs already exist, skipping: $(basename "$trace_dir")"
    return 0
  fi

  echo "Generating features for $(basename "$trace_dir")"
  
  cd "$PEREGRINE_ROOT" && python3 ronamol/python/gen_features.py "$trace_file"
}

export -f run_cache_latency
export -f run_bp_rate
export -f run_features

# Function to generate all cache latency combinations for a trace
generate_cache_combinations() {
  local trace_file="$1"
  
  for l1i_size in "${L1_KB[@]}"; do
    for l1d_size in "${L1_KB[@]}"; do
      for l2_size in "${L2_KB[@]}"; do
        echo "$trace_file $l1i_size $l1d_size $l2_size"
      done
    done
  done
}

# Function to generate all branch predictor combinations for a trace
generate_bp_combinations() {
  local trace_file="$1"
  
  for bp_type in "${BP_TYPES[@]}"; do
    echo "$trace_file $bp_type"
  done
}

# Find all trace.csv files in TRACE_DIR subdirectories
trace_files=()
while IFS= read -r -d '' trace_file; do
  trace_files+=("$trace_file")
done < <(find "$TRACE_DIR" -name "trace.csv" -type f -print0)

if [[ ${#trace_files[@]} -eq 0 ]]; then
  echo "No trace.csv files found in $TRACE_DIR" >&2
  exit 1
fi

echo "Found ${#trace_files[@]} trace files in $TRACE_DIR"

# Calculate total number of jobs
total_cache_jobs=$((${#trace_files[@]} * ${#L1_KB[@]} * ${#L1_KB[@]} * ${#L2_KB[@]}))
total_bp_jobs=$((${#trace_files[@]} * ${#BP_TYPES[@]}))
total_feature_jobs=${#trace_files[@]}
total_jobs=$((total_cache_jobs + total_bp_jobs + total_feature_jobs))

echo "Total jobs to run:"
echo "  Cache latency jobs: $total_cache_jobs (${#trace_files[@]} traces × ${#L1_KB[@]}×${#L1_KB[@]}×${#L2_KB[@]} configurations)"
echo "  Branch predictor jobs: $total_bp_jobs (${#trace_files[@]} traces × ${#BP_TYPES[@]} types)"
echo "  Feature generation jobs: $total_feature_jobs (${#trace_files[@]} traces)"
echo "  Total: $total_jobs jobs"

echo ""
echo "Starting parallel execution at $(date '+%Y-%m-%d %H:%M:%S %Z')"
SWEEP_START_EPOCH=$(date +%s)

# Generate and run cache latency jobs in parallel
echo "Running cache latency simulations..."
for trace_file in "${trace_files[@]}"; do
  generate_cache_combinations "$trace_file"
done | parallel -j "$(nproc)" --colsep ' ' \
  --env PEREGRINE_ROOT --env run_cache_latency \
  run_cache_latency {1} {2} {3} {4}

# Generate and run branch predictor jobs in parallel
echo "Running branch predictor simulations..."
for trace_file in "${trace_files[@]}"; do
  generate_bp_combinations "$trace_file"
done | parallel -j "$(nproc)" --colsep ' ' \
  --env PEREGRINE_ROOT --env run_bp_rate \
  run_bp_rate {1} {2}

# Generate features for all traces in parallel
echo "Generating features..."
printf '%s\n' "${trace_files[@]}" | parallel -j "$(nproc)" \
  --env PEREGRINE_ROOT --env run_features \
  run_features {}

SWEEP_END_EPOCH=$(date +%s)
SWEEP_ELAPSED=$((SWEEP_END_EPOCH - SWEEP_START_EPOCH))
SWEEP_ELAPSED_H=$((SWEEP_ELAPSED / 3600))
SWEEP_ELAPSED_M=$(((SWEEP_ELAPSED % 3600) / 60))
SWEEP_ELAPSED_S=$((SWEEP_ELAPSED % 60))

echo ""
echo "Sweep finished at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Total wall time: ${SWEEP_ELAPSED}s (${SWEEP_ELAPSED_H}h ${SWEEP_ELAPSED_M}m ${SWEEP_ELAPSED_S}s)"
