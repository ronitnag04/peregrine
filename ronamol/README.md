## Ronamol — simple analytical model

### What it produces

- **Program vector**: compact "program pressure" features derived from the
  trace (instruction mix, control-flow, dependency structure, memory behavior).
- **Optional per-cache-config latency summaries**: reads the cache sidecars
  produced by `peregrine/gen_cache_latencies.py`.
- **Branch predictor rates**: reads the BP sidecar produced by
  `peregrine/gen_bp_rates.py`.

### Prerequisites

Before running the analytical model, you need to generate cache latency and branch predictor sidecar files for each trace. Run this from the root directory of the peregrine repo
```bash
for f in ronamol/traces/*/trace.csv; do python3 gen_bp_rates.py "$f"; done
```

```bash
for f in ronamol/traces/*/trace.csv; do python3 gen_cache_latencies.py "$f"; done
```

### Run

Generate the program features and cache summary features. Run these commands from the `ronamol` directory.
```bash
cd ronamol
for f in traces/*/trace.csv; do python3 python/gen_features.py "$f"; done
```

Outputs are written next to the trace by default:
`<trace_dir>/ronamol/program_features.json` (and optionally
`<trace_dir>/ronamol/cache_latency_summary.csv`).

### Analyze Results

You can optionally analyze the results of the ronamol model with the `python/analyze_ronamol.py` script.
```bash
python3 python/analyze_ronamol.py -i traces/ -s sweep/ofat_sweep_results.csv
```

### Build training CSV from a gem5 sweep

Finally, build the training matrix to use with the ML Model.
```bash
python3 python/build_training_csv.py --sweep-results sweep/sweep_results.csv -o training_data.csv
```

This joins each sweep row with:
- `<traces_root>/<benchmark>/ronamol/program_features.json` (by `benchmark`)
- `<traces_root>/<benchmark>/ronamol/cache_latency_summary.csv` (by cache sizes in the sweep row parameters)
- `<traces_root>/<benchmark>/ronamol/trace_bp.json` (by branch predictor type in the sweep row parameters)

### Build prediction CSV for ML Model inference tests

Use this script to generate data points for the ML model to predict on to test the inference speed of the model over the design space.
```bash
python3 python/gen_prediction_sweep.py
```