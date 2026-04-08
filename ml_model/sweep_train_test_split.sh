#!/bin/bash

# Sweep script for different train-test split sizes
# Calls train.py with different test-size values

set -euo pipefail

DATASET_PATH="training_data/ronamol_spec_training_data.csv"
PYTHON_SCRIPT="train.py"

# Array of test sizes to sweep over (as fractions)
TEST_SIZES=(0.1 0.15 0.2 0.25 0.3 0.35 0.4)

echo "Starting train-test split sweep with dataset: $DATASET_PATH"
echo "Test sizes to sweep: ${TEST_SIZES[*]}"
echo "=========================================="

# Create results directory
mkdir -p sweep_results
SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="sweep_results/sweep_${SWEEP_TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

# Log file for the sweep
SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Dataset: $DATASET_PATH" >> "$SWEEP_LOG"
echo "Test sizes: ${TEST_SIZES[*]}" >> "$SWEEP_LOG"
echo "========================================" >> "$SWEEP_LOG"

for test_size in "${TEST_SIZES[@]}"; do
    echo ""
    echo "Running training with test size: $test_size"
    echo "Time: $(date)"
    
    # Log to sweep log
    echo "" >> "$SWEEP_LOG"
    echo "Test size: $test_size - Started at $(date)" >> "$SWEEP_LOG"
    
    # Create subdirectory for this run; train.py writes checkpoint, scaler, and metrics here
    RUN_DIR="$SWEEP_DIR/test_size_${test_size}"
    mkdir -p "$RUN_DIR"
    OUTPUT_DIR="$RUN_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
    SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
    METRICS_PATH="$OUTPUT_DIR/metrics.json"

    # Run training and capture output
    if python "$PYTHON_SCRIPT" -d "$DATASET_PATH" --test-size "$test_size" --output-dir "$OUTPUT_DIR" > "$RUN_DIR/training_output.txt" 2>&1; then
        echo "✓ Training completed successfully for test size $test_size"
        echo "Test size: $test_size - COMPLETED at $(date)" >> "$SWEEP_LOG"
    else
        echo "✗ Training failed for test size $test_size"
        echo "Test size: $test_size - FAILED at $(date)" >> "$SWEEP_LOG"
    fi
done

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Results saved in: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

# Create a summary of results
SUMMARY_FILE="$SWEEP_DIR/summary.txt"
echo "Train-Test Split Sweep Summary" > "$SUMMARY_FILE"
echo "Generated at: $(date)" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET_PATH" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for test_size in "${TEST_SIZES[@]}"; do
    RUN_DIR="$SWEEP_DIR/test_size_${test_size}"
    OUTPUT_DIR="$RUN_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
    SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
    METRICS_PATH="$OUTPUT_DIR/metrics.json"
    if [ -f "$RUN_DIR/training_output.txt" ]; then
        echo "Test Size: $test_size" >> "$SUMMARY_FILE"
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