# `sweep_to_training.py` — Context and Changes

This document records every file that was created or modified to make
`sweep_to_training.py` work, and explains why each change was necessary.

---

## What the script does

`sweep_to_training.py` converts the output of a **gem5 parameter sweep**
directly into an anamol training DataFrame, bypassing the normal anamol
workflow (simulate → lookup → training) when the simulation has already
been run.

The gem5 sweep produces a `sweep_results.csv` where each row is one
`(benchmark, architecture config, measured CPI)` triple. For each row the
script:

1. Parses the gem5 parameter names and value formats into an anamol `Config`
2. Looks up the precomputed throughput features for that config from a
   `ThroughputLookupTable` (built once per benchmark from existing `.npy` files)
3. Calls `generate_training_matrix` to assemble the full feature vector
4. Attaches the CPI label
5. Collects all rows into a single DataFrame and saves it

Two modes:
- **Precomputed mode** (`--precomputed-dir`): anamol has already been run; only
  the lookup table needs to be built (or loaded from cache). All rows are
  parallelised across threads.
- **Single-config mode** (default, no sweep): runs the anamol binary for each
  row individually (sequential, because spawning many subprocesses is wasteful).

---

## New files

### `python/sweep_to_training.py`

The main script. Depends on everything listed below.

Key internal structure:

| Function | Purpose |
|---|---|
| `_build_gem5_mappings()` | Reads `registry.PARAMS` to build gem5-column → Config-field maps at import time |
| `parse_gem5_row(row)` | Converts one gem5 CSV row into a `models.Config` |
| `_config_to_hash(config)` | MD5 hash of all enabled param values — used as a cache directory key |
| `_get_config_idx(trace_dir, config)` | Looks up the cache-config index in `trace_configs.json` for latency-dependent features |
| `_run_anamol(...)` | Spawns the anamol C++ binary for single-config runs |
| `_compute_thr_features_from_output(...)` | Reads `.npy` files directly (no lookup table) for single-config mode |
| `_process_single_row(...)` | Per-row worker dispatched to `ThreadPoolExecutor` in precomputed mode |
| `_process_with_sweep(...)` | Precomputed/sweep mode: builds lookup tables then parallelises row processing |
| `process_sweep_csv(...)` | Top-level entry point |

---

### `python/registry.py`

Python-side reader for `registry.yaml`. Loaded once at import time and
provides the authoritative lists of resources and params for all Python scripts.

`sweep_to_training.py` uses:
- `registry.PARAMS` / `registry.ENABLED_PARAMS` — to build the gem5 → Config
  field mapping automatically (so new params registered in YAML are picked up
  without touching the script)
- `registry.ENABLED_RESOURCES` / `registry.LATENCY_DEPENDENT_RESOURCES` —
  to iterate over resources when computing features directly from `.npy` files

Without `registry.py`, the gem5 → Config mapping would have to be hand-coded
and kept in sync manually.

---

### `python/gen_registry.py`

Code generator. Reads `registry.yaml` and writes:

| Output file | Contents |
|---|---|
| `include/resources_gen.h` | `Resource` enum |
| `include/params_gen.h` | `ParamType` enum, `PARAM_RANGES`, `ParamSweep` |
| `include/resource_registry.h` | `RESOURCE_REGISTRY` — wires each resource to its `get_thr_*` function |
| `include/models_decl_gen.h` | `get_thr_*` forward declarations |
| `python/config_gen.py` | `Config` dataclass with defaults from registry |

Run automatically by `make` when `registry.yaml` changes.

---

### `registry.yaml`

Single source of truth for all resources and params. Defines each param's
name, sweep range, default, and gem5 column name (`name_gem5`). The
`name_gem5` field is what `sweep_to_training.py` reads to auto-build its
mapping — adding a new param to the YAML automatically makes it flow through
to training without code changes.

---

### `python/consolidate_training.py`

Utility script added alongside the sweep pipeline. Reads all per-trace `.pkl`
files from the `training/` directory and concatenates them into a single matrix
(`training/all_traces.pkl`). Called via `make consolidate.run`.

---

### `gen_cache_latencies.py` (repo root)

Runs the **cache simulation** across all L1I × L1D × L2 cache parameter
combinations for a trace (100 configs), producing:
- `<trace_dir>/trace_latencies.npy` — shape `(100, N_instrs, 2)` with
  per-instruction fetch and exec latencies for each cache config
- `<trace_dir>/trace_configs.json` — the config grid so C++ can look up the
  config index

The latency `.npy` is fed to anamol via `-l`, enabling latency-dependent
resources (ROB, load queue, etc.) to vary by cache config. This is what allows
`_get_config_idx()` in `sweep_to_training.py` to select the right per-cache-config
throughput subdir.

Run via: `make gen_latencies.run TRACE_CSV=<path>` or
`python gen_cache_latencies.py <trace.csv>`

---

### `gen_bp_rates.py` (repo root)

Runs the **branch predictor simulation** for a trace, producing:
- `<trace_dir>/trace_bp.json` — `{"local": <float>, "tage": <float>}` with
  per-predictor misprediction rates in [0, 1]

BP simulation is independent of cache configs and only needs to be run once
per trace. `sweep_to_training.py` reads `trace_bp.json` via `_load_bp_rate()`
to populate `config.misprediction_percent` before generating training features.

Run via: `make gen_bp.run TRACE_CSV=<path>` or
`python gen_bp_rates.py <trace.csv>`

---

## Modified files

### `Makefile`

The Makefile grew significantly to support the full pipeline. Changes relevant
to `sweep_to_training.py`:

**Variables added:**

| Variable | Purpose |
|---|---|
| `LATENCIES_NPY` | Path to `.npy` latency file; passed to anamol with `-l` |
| `CONFIGS_JSON` | Path to configs JSON; passed to `build_throughput_lookup.py` |
| `TRACE_DIR` | Shorthand to derive `TRACE_CSV`, `LATENCIES_NPY`, `CONFIGS_JSON` from one folder |
| `OUTPUT_DIR` | Per-trace output directory `output/<trace_name>/` |
| `LOOKUP_FILE` | Path to the per-trace `throughput_lookup.pkl` |
| `TRAINING_DIR` / `TRAINING_OUT` | Training output locations |
| `SWEEP_CSV` | gem5 sweep results CSV |
| `BENCHMARKS` | Comma-separated benchmark whitelist |
| `SWEEP_DO_SWEEP` | Enable full param sweep per benchmark (vs. per-row single-config) |
| `SWEEP_FORMAT` | Output format: `pkl` or `csv` |
| `SWEEP_OUT` | Output path for training DataFrame |
| `SWEEP_PRECOMPUTED_DIR` | Directory of already-computed anamol outputs; skips the binary |
| `SWEEP_WORKERS` | Number of parallel threads for `sweep_training.run` |

**Targets added:**

| Target | What it does |
|---|---|
| `lookup.run` | Runs `build_throughput_lookup.py` to build/save the lookup table |
| `training.run` | Runs `gen_training_data.py` for random or specific configs |
| `pipeline.run` | `main.run` → `lookup.run` → `training.run` for one trace |
| `all_traces.run` | Runs `pipeline.run` for every `*_with_latency.csv` in `traces/` |
| `consolidate.run` | Runs `consolidate_training.py` to merge all per-trace `.pkl` files |
| `sweep_training.run` | Runs `sweep_to_training.py` |
| `clean-gen` | Deletes all generated headers and `config_gen.py` |

**Other build changes:**
- Added `REGISTRY_CONFIG`, `GENERATED_HEADERS`, `GENERATED_PY` and a
  recipe that auto-runs `gen_registry.py` when `registry.yaml` changes
- All object file rules now depend on `$(GENERATED_HEADERS)` so a YAML change
  triggers a full C++ rebuild
- `main.run` now passes `-o $(OUTPUT_DIR)` and optionally `-l $(LATENCIES_NPY)`

---

### `src/main.cpp`

Added three new CLI flags that `_run_anamol()` in `sweep_to_training.py`
constructs and passes:

| Flag | Purpose |
|---|---|
| `-o <dir>` | Output directory (previously hardcoded to `output/`) |
| `-l <path>` | Path to `trace_latencies.npy` for latency-dependent resources |
| `--config-json <json>` | Run in single-config mode instead of full sweep |

The `--config-json` flag is what makes the single-config path in
`sweep_to_training.py` possible — anamol normally sweeps all parameter
combinations, but for the per-row case we only need one.

---

### `python/build_throughput_lookup.py`

Updated to support **latency-dependent resources**. When `--configs-json` is
supplied, the lookup table keys for latency-dependent resources include the
cache config index:

```
(resource, (cache_config_idx, *param_combo)) -> 101-dim feature vector
```

The latency-dependent subdir layout (`output/<trace>/config_NNNN/thr_*.npy`)
was introduced alongside this change. `sweep_to_training.py` uses
`_get_config_idx()` to resolve which subdir to read from when computing
features directly from `.npy` files (single-config mode).

---

### `python/gen_training_data.py`

Updated to accept:
- `include_latency_features: bool` — whether to include ROB latency features
  (passed as `True` by `sweep_to_training.py`)
- `output_dir: str` — path to the per-benchmark anamol output directory
  (needed because the sweep pipeline uses per-benchmark subdirectories rather
  than a single `output/`)

`sweep_to_training.py` calls `generate_training_matrix` with both of these.

---

### `python/models.py`

Updated to re-export from `registry.py` and `config_gen.py` rather than
defining resources and params manually. The `Config` dataclass is now generated
(`python/config_gen.py`) from `registry.yaml` instead of being hand-written.

`sweep_to_training.py` imports `models.Config` directly; having it driven by
the registry means `parse_gem5_row` just needs `registry.PARAMS` to find every
field name — no separate sync required.

---

## Data flow in precomputed mode

```
gem5 sweep_results.csv
        │
        ▼
sweep_to_training.py
  ├─ parse_gem5_row()          registry.PARAMS → Config
  │
  ├─ _process_with_sweep()
  │    ├─ ThroughputLookupTable.load()    precomputed throughput_lookup.pkl
  │    │   (or build from thr_*.npy via build_throughput_lookup.py)
  │    │
  │    └─ ThreadPoolExecutor
  │         └─ _process_single_row() × N rows (parallel)
  │              └─ generate_training_matrix()
  │                   ├─ lookup.get_config_features()   throughput features
  │                   ├─ compute_pipeline_stall_features()  trace stats
  │                   ├─ compute_rob_latency_features()     latency .npy
  │                   └─ get_config_scalar_features()       arch encoding
  │
  └─ pd.concat() → save .pkl / .csv
```
