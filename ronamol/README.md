## Ronamol — simple analytical model

### What it produces

- **Program vector**: compact "program pressure" features derived from the
  trace (instruction mix, control-flow, dependency structure, memory behavior).
- **Cache latency summaries**: per-cache-config latency summaries from cache simulations.
- **Branch predictor rates**: misprediction rates from branch predictor simulations.

### Prerequisites

This workflow assumes you have:
- Trace files in `trace.csv` format in subdirectories
- The `peregrine` repository with `gen_cache_latency.py`, `gen_bp_rate.py`, and related scripts
- GNU parallel installed (`apt install parallel`)

### Generate Training Data

The training data generation process has been streamlined into two main steps:

#### Step 1: Run the Complete Sweep

Run the `sweep_traces.sh` script from the peregrine root directory to generate all cache latencies, branch predictor rates, and ronamol features in parallel:

```bash
cd /path/to/peregrine
export TRACE_DIR="/path/to/your/traces"  # Directory containing trace subdirs with trace.csv files
./sweep_traces.sh
```

This single script will:
1. Generate cache latencies for all 100 configurations (5×5×4 L1I×L1D×L2 combinations)
2. Generate branch predictor rates for all types (local, tage)
3. Generate ronamol features (program features + summaries)

The script runs all operations in parallel across available CPU cores and creates the following structure for each trace:
```
<trace_dir>/
├── trace.csv
├── cache_latencies/
│   ├── l1i_16_l1d_16_l2_512_cache_latencies.npy
│   ├── l1i_32_l1d_32_l2_1024_cache_latencies.npy
│   └── ... (other cache configurations)
├── bp_rates/
│   ├── local_bp_rate.npy
│   └── tage_bp_rate.npy
└── ronamol/
    ├── program_features.json
    ├── cache_latency_summary.csv
    └── bp_rates_summary.csv
```

#### Step 2: Build the Training CSV

Once the sweep completes, build the final training matrix from a gem5 sweep results file:

```bash
cd ronamol
python3 python/build_training_csv.py \
    --sweep-results sweep/sweep_results.csv \
    --traces-root /path/to/your/traces \
    -o training_data.csv
```

This joins each sweep row with:
- `<traces_root>/<benchmark>/ronamol/program_features.json` (by `benchmark`)
- `<traces_root>/<benchmark>/ronamol/cache_latency_summary.csv` (by cache sizes in the sweep row parameters)  
- `<traces_root>/<benchmark>/ronamol/bp_rates_summary.csv` (by branch predictor type in the sweep row parameters)

### Alternative: Manual Step-by-Step Process

If you prefer to run individual steps manually or need more control:

#### Generate cache latencies and branch predictor rates:
```bash
cd /path/to/peregrine
for trace_file in "$TRACE_DIR"/*/trace.csv; do
    python3 gen_cache_latency.py --trace "$trace_file" --l1i-size 32 --l1d-size 32 --l2-size 1024
    python3 gen_bp_rate.py --trace "$trace_file" --branch-predictor local
    # ... repeat for all configurations
done
```

#### Generate ronamol features:
```bash
cd ronamol  
for trace_file in "$TRACE_DIR"/*/trace.csv; do
    python3 python/gen_features.py "$trace_file"
done
```

### Analyze Results

You can optionally analyze the results of the ronamol model with the `python/analyze_ronamol.py` script.
```bash
python3 python/analyze_ronamol.py -i /path/to/traces -s sweep/ofat_sweep_results.csv
```

### Build prediction CSV for ML Model inference tests

Use this script to generate data points for the ML model to predict on to test the inference speed of the model over the design space.
```bash
python3 python/gen_prediction_sweep.py
```