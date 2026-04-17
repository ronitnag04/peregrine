#!/bin/bash

# Sweep script for different out-of-distribution regions and hw configs

set -euo pipefail

DATASET_PATH="training_data/ronamol_spec_training_data_v2.csv"
PYTHON_SCRIPT="train_split_configs_regions.py"

test_regions_splits=(0.00 0.0625 0.125 0.1875 0.25)
test_configs_splits=(0.00 0.0625 0.125 0.1875 0.25)

echo "Starting out-of-distribution regions and configs sweep with dataset: $DATASET_PATH"
echo "Test regions splits: ${test_regions_splits[*]}"
echo "Test configs splits: ${test_configs_splits[*]}"
echo "=========================================="

# Create results directory
mkdir -p sweep_results
SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# SWEEP_DIR="sweep_results/sweep_${SWEEP_TIMESTAMP}"
SWEEP_DIR="sweep_results/ood_regions_configs_spec_v2"
mkdir -p "$SWEEP_DIR"

# Log file for the sweep
SWEEP_LOG="$SWEEP_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Dataset: $DATASET_PATH" >> "$SWEEP_LOG"
echo "Test regions splits: ${test_regions_splits[*]}" >> "$SWEEP_LOG"
echo "Test configs splits: ${test_configs_splits[*]}" >> "$SWEEP_LOG"
echo "========================================" >> "$SWEEP_LOG"

for test_regions_split in "${test_regions_splits[@]}"; do
    for test_configs_split in "${test_configs_splits[@]}"; do
        if (( $(echo "$test_regions_split == 0.00" | bc -l) )) && (( $(echo "$test_configs_split == 0.00" | bc -l) )); then   
            echo "Skipping test_regions_split=$test_regions_split test_configs_split=$test_configs_split"
            continue
        fi
        echo ""
        echo "Running training with test_regions_split=$test_regions_split test_configs_split=$test_configs_split"
        echo "Time: $(date)"
        
        # Log to sweep log
        echo "" >> "$SWEEP_LOG"
        echo "Test regions split: $test_regions_split - Test configs split: $test_configs_split - Started at $(date)" >> "$SWEEP_LOG"
        
        # Create subdirectory for this run; train.py writes checkpoint, scaler, and metrics here
        RUN_DIR="$SWEEP_DIR/ood_regions_${test_regions_split}_configs_${test_configs_split}"
        mkdir -p "$RUN_DIR"
        OUTPUT_DIR="$RUN_DIR"
        CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
        SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
        METRICS_PATH="$OUTPUT_DIR/metrics.json"

        # Run training and capture output
        if python "$PYTHON_SCRIPT" -d "$DATASET_PATH" --test-regions "$test_regions_split" --test-configs "$test_configs_split" --output-dir "$OUTPUT_DIR" > "$RUN_DIR/training_output.txt" 2>&1; then
            echo "✓ Training completed successfully for test_regions_split=$test_regions_split test_configs_split=$test_configs_split"
            echo "Test regions split: $test_regions_split - Test configs split: $test_configs_split - COMPLETED at $(date)" >> "$SWEEP_LOG"
        else
            echo "✗ Training failed for test_regions_split=$test_regions_split test_configs_split=$test_configs_split"
            echo "Test regions split: $test_regions_split - Test configs split: $test_configs_split - FAILED at $(date)" >> "$SWEEP_LOG"
        fi
    done
done

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Results saved in: $SWEEP_DIR"
echo "Sweep completed at $(date)" >> "$SWEEP_LOG"

# Create a summary of results
SUMMARY_FILE="$SWEEP_DIR/summary.txt"
echo "Out-of-distribution regions and configs Sweep Summary" > "$SUMMARY_FILE"
echo "Generated at: $(date)" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET_PATH" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for test_regions_split in "${test_regions_splits[@]}"; do
    for test_configs_split in "${test_configs_splits[@]}"; do
        RUN_DIR="$SWEEP_DIR/ood_regions_${test_regions_split}_configs_${test_configs_split}"
        OUTPUT_DIR="$RUN_DIR"
        CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint.pt"
        SCALER_PATH="$OUTPUT_DIR/scaler.joblib"
        METRICS_PATH="$OUTPUT_DIR/metrics.json"
        if [ -f "$RUN_DIR/training_output.txt" ]; then
            echo "Test regions split: $test_regions_split - Test configs split: $test_configs_split" >> "$SUMMARY_FILE"
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
done

echo "Summary created: $SUMMARY_FILE"