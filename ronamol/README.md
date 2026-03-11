## Ronamol — simple analytical model

### What it produces

- **Program vector**: compact "program pressure" features derived from the
  trace (instruction mix, control-flow, dependency structure, memory behavior).
- **Optional per-cache-config latency summaries**: reads the cache sidecars
  produced by `peregrine/gen_cache_latencies.py`.
- **Branch predictor rates**: reads the BP sidecar produced by
  `peregrine/gen_bp_rates.py`.

### Run

```bash
python3 peregrine/ronamol/python/gen_features.py peregrine/anamol/traces/collatz/trace.csv --no-cache-lat-summary
```

Outputs are written next to the trace by default:
`<trace_dir>/ronamol/program_features.json` (and optionally
`<trace_dir>/ronamol/cache_latency_summary.csv`).

### Build training CSV from a gem5 sweep

```bash
python3 peregrine/ronamol/python/build_training_csv.py \
  --sweep-results /path/to/sweep_results.csv \
  -o /path/to/training.csv
```

This joins each sweep row with:
- `<traces_root>/<benchmark>/ronamol/program_features.json` (by `benchmark`)
- `<traces_root>/<benchmark>/ronamol/cache_latency_summary.csv` (by cache sizes)
- `<traces_root>/<benchmark>/ronamol/trace_bp.json` (by branch predictor type)

