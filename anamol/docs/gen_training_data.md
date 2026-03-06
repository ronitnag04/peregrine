# Generating Training Data

All `make` commands run from `anamol/`.
```bash
cd anamol/
```

---

## Method A — Analytical model + lookup table

### 1. Build

```bash
make
```

### 2. Run analytical model

```bash
make main.run TRACE_DIR=traces/foo WINDOW_SIZE=400
```

### 3. Build lookup table

```bash
make lookup.run TRACE_DIR=traces/foo
```

### 4. Generate training data

```bash
# N random configs:
make training.run TRACE_DIR=traces/foo RANDOM_CONFIGS=1000 SEED=42

# Specific config:
make training.run TRACE_DIR=traces/foo 'CONFIG_JSON={"rob_size":256,"load_queue_size":128}'
```

Output: `training/foo.pkl`

### Full pipeline (steps 2–4)

```bash
make pipeline.run TRACE_DIR=traces/foo WINDOW_SIZE=400 RANDOM_CONFIGS=1000 SEED=42
```

### Consolidate multiple traces

```bash
make consolidate.run   # training/*.pkl → training/all_traces.pkl
```

---

## Method B — gem5 sweep

Use this when you already have gem5 sweep results.

### 1. Generate BP rates and cache latencies (once per trace)

```bash
# Repeat for each benchmark:
make gen_bp.run TRACE_CSV=traces/foo-pin/trace.csv
make gen_latencies.run TRACE_CSV=traces/foo-pin/trace.csv
```

### 2. Build
```bash
make
```

### 3. Run analytical model per benchmark

```bash
# Repeat for each benchmark:
make main.run TRACE_DIR=traces/foo-pin
```

### 4. Generate training data from gem5 sweep

```bash
make sweep_training.run \
    SWEEP_PRECOMPUTED_DIR=output \   # point to anamol output from step 3
    BENCHMARKS=foo,bar \             # optional: filter to specific benchmarks
    SWEEP_TRACES_DIR=traces \        # optional: where benchmark folders with trace CSVs live (default: traces/)
    SWEEP_FORMAT=csv \               # optional: pkl (default) or csv
    SWEEP_OUT=training/my_training.csv  # optional: output path
```

`SWEEP_CSV` defaults to the latest sweep dir in the Makefile — override if using a different one:
```bash
make sweep_training.run SWEEP_CSV=sim_sweeps/my_sweep/sweep_results.csv ...
```

| Variable | Default | Description |
|---|---|---|
| `SWEEP_CSV` | `sim_sweeps/sim_sweep_02_24_2026/sweep_results.csv` | gem5 sweep results |
| `SWEEP_PRECOMPUTED_DIR` | _(empty)_ | Anamol output dir from step 3 (organized as `<dir>/<benchmark>/`) |
| `BENCHMARKS` | _(all in CSV)_ | Comma-separated benchmark whitelist |
| `SWEEP_TRACES_DIR` | `traces` | Directory containing trace CSVs |
| `SWEEP_FORMAT` | `pkl` | Output format: `pkl` or `csv` |
| `SWEEP_OUT` | `training/sweep_training.<format>` | Output file path |
| `SWEEP_CACHE_DIR` | `.cache/sweep_out` | Intermediate output directory |
| `SWEEP_WORKERS` | _(cpu count)_ | Parallel worker threads |
