#!/bin/bash

# Sweep script: train one model per benchmark (train/test split within that benchmark only).
# Calls train_single_bmark.py once per benchmark name.

set -euo pipefail

DATASET_PATH="training_data/ronamol_spec_training_data_v3.csv"
PYTHON_SCRIPT="train_single_bmark.py"

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r")

echo "Starting per-benchmark single-model sweep with dataset: $DATASET_PATH"
echo "Benchmarks to sweep: ${BENCHMARKS[*]}"
echo "=========================================="

# Create results directory
mkdir -p sweep_results
# SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# SWEEP_DIR="sweep_results/sweep_${SWEEP_TIMESTAMP}"
SWEEP_DIR="sweep_results/spec_v3_single_bmark"
mkdir -p "$SWEEP_DIR"

# Log file for the sweep
SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Dataset: $DATASET_PATH" >> "$SWEEP_LOG"
echo "Benchmarks: ${BENCHMARKS[*]}" >> "$SWEEP_LOG"
echo "========================================" >> "$SWEEP_LOG"

for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "Running train_single_bmark for benchmark: $benchmark"
    echo "Time: $(date)"

    # Log to sweep log
    echo "" >> "$SWEEP_LOG"
    echo "Benchmark: $benchmark - Started at $(date)" >> "$SWEEP_LOG"

    # Create subdirectory for this run; train_single_bmark.py writes checkpoint, scaler, and metrics here
    RUN_DIR="$SWEEP_DIR/${benchmark}"
    mkdir -p "$RUN_DIR"
    OUTPUT_DIR="$RUN_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
    SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
    METRICS_PATH="$OUTPUT_DIR/metrics.json"

    # Run training and capture output
    if python "$PYTHON_SCRIPT" -d "$DATASET_PATH" --benchmark "$benchmark" --output-dir "$OUTPUT_DIR" > "$RUN_DIR/training_output.txt" 2>&1; then
        echo "✓ Training completed successfully for benchmark: $benchmark"
        echo "Benchmark: $benchmark - COMPLETED at $(date)" >> "$SWEEP_LOG"
    else
        echo "✗ Training failed for benchmark: $benchmark"
        echo "Benchmark: $benchmark - FAILED at $(date)" >> "$SWEEP_LOG"
    fi
done

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Results saved in: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

# Create a summary of results
SUMMARY_FILE="$SWEEP_DIR/summary.txt"
echo "Per-benchmark single-model sweep summary" > "$SUMMARY_FILE"
echo "Generated at: $(date)" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET_PATH" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for benchmark in "${BENCHMARKS[@]}"; do
    RUN_DIR="$SWEEP_DIR/${benchmark}"
    OUTPUT_DIR="$RUN_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
    SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
    METRICS_PATH="$OUTPUT_DIR/metrics.json"
    if [ -f "$RUN_DIR/training_output.txt" ]; then
        echo "Benchmark: $benchmark" >> "$SUMMARY_FILE"
        # Extract final percent error from training output if available
        if grep -q "Percent Error" "$RUN_DIR/training_output.txt"; then
            FINAL_METRICS=$(tail -20 "$RUN_DIR/training_output.txt" | grep "Percent Error" | tail -1)
            # Extract just the percent error value from the line
            PERCENT_ERROR=$(echo "$FINAL_METRICS" | grep -o "Percent Error: [0-9]*\.[0-9]*%" | cut -d' ' -f3)
            echo "  Final Percent Error: $PERCENT_ERROR" >> "$SUMMARY_FILE"
        fi
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "Summary created: $SUMMARY_FILE"
