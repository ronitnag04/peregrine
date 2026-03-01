# Anamol — Workflow Reference

Anamol is the analytical modeling component of Peregrine. It simulates microarchitectural resource throughput distributions over sliding windows of a dynamic instruction trace, then assembles those distributions into feature vectors for training a CPI prediction model.

---

## Directory structure

```
anamol/
├── registry.yaml            # Single source of truth — all params and resources
├── Makefile
│
├── include/                 # C++ headers
│   ├── param_types.h        # Manual: StepType, ParamRange (infrastructure)
│   ├── params.h             # Umbrella: #includes params_gen.h
│   ├── models.h             # Manual: ResourceEntry, ThrVec, ThrFunc; #includes models_decl_gen.h
│   ├── instr.h              # Instruction struct
│   ├── parser.h
│   ├── resources_gen.h      # GENERATED: Resource enum
│   ├── params_gen.h         # GENERATED: ParamType enum, PARAM_RANGES, ParamSweep
│   ├── resource_registry.h  # GENERATED: RESOURCE_REGISTRY vector
│   └── models_decl_gen.h    # GENERATED: get_thr_* function declarations
│
├── src/
│   ├── models.cpp           # All get_thr_* implementations + get_throughput()
│   ├── main.cpp             # Entry point: parse trace → run models → export
│   ├── npy_reader.h         # Minimal numpy v1.0 reader for latency .npy files
│   ├── parser.cpp           # CSV trace parser
│   └── convert_trace.cpp    # Trace format conversion utility
│
├── tests/
│   ├── test_params.cpp      # Validates param sweep counts
│   └── test_models.cpp      # Runs all models on a trace, prints throughputs
│
├── python/
│   ├── registry.py          # Reads registry.yaml at import time — Python source of truth
│   ├── models.py            # Resource enum + re-exports from registry and config_gen
│   ├── config_gen.py        # GENERATED: Config dataclass with defaults from registry
│   ├── utils.py             # CDF feature computation, .npy loading
│   ├── gen_registry.py      # Generator: registry.yaml → C++ headers + config_gen.py
│   ├── build_throughput_lookup.py  # Builds lookup table from output/*.npy
│   ├── gen_training_data.py        # Assembles full feature matrix for training
│   ├── consolidate_training.py     # Merges training data across multiple traces
│   ├── visualize.py                # Plotting utilities
│   └── example_plot_thr.py         # Example: plot throughput distributions
│
├── traces/                  # Input dynamic instruction traces
│   └── foo/                 # One folder per trace (recommended)
│       ├── trace.csv            # Raw trace
│       ├── trace_latencies.npy  # Produced by annotate_trace.py — (N_configs, N_instrs, 2) uint16
│       └── trace_configs.json   # Config index map produced alongside the .npy
└── output/                  # Analytical model output
    ├── <stem>/
    │   ├── thr_alu_issue.npy        # Latency-independent resources (written once)
    │   ├── thr_fp_issue.npy
    │   ├── ...
    │   ├── config_0000/             # Per-cache-config output
    │   │   ├── thr_rob.npy          # Latency-dependent resources
    │   │   ├── thr_load_queue.npy
    │   │   ├── rob_latency_*.npy
    │   │   └── ...
    │   └── config_0199/
    │       └── ...
```

---

## Single source of truth: `registry.yaml`

`registry.yaml` is the only place where resources and parameters are defined. Everything else is derived from it.

**Params** define the microarchitectural knobs:
```yaml
params:
  - name: rob_size      # Python Config field name; C++ enum: ROB_SIZE
    min: 1
    max: 1024
    step: linear        # "base2", "linear", or a discrete list, e.g. [0, 4]
    default: 128
    enabled: true
```

**Resources** define the analytical models and which params drive their sweep:
```yaml
resources:
  - name: rob                        # Python key; C++ enum: ROB; function: get_thr_rob; file: thr_rob.npy
    params: [rob_size]               # 1 or 2 params
    enabled: true
    latency_dependent: [exe]         # calls resp_cycle() → exe_latency; re-run per cache config

  - name: alu_issue
    params: [alu_issue_width]
    enabled: true
    latency_dependent: []            # instruction count only; computed once

  - name: icache_fills
    params: [max_icache_fills]
    enabled: true
    latency_dependent: [fetch]       # uses instr.fetch_latency directly

  - name: load_ls_pipes_lower
    params: [num_ls_pipes, num_load_pipes]   # 2-param resource
    enabled: true
    latency_dependent: []

  - name: fetch_buffers
    params: [num_fetch_buffers]
    enabled: false                   # skipped by C++ and Python
    latency_dependent: [fetch]       # uses instr.fetch_latency directly
```

`latency_dependent` is a list of latency types the resource's model reads:
- `[exe]` — calls `resp_cycle()` → `instr.exe_latency` (ROB, LQ, SQ)
- `[fetch]` — reads `instr.fetch_latency` directly (icache_fills, fetch_buffers)
- `[exe, fetch]` — depends on both
- `[]` — instruction counts only; computed once, shared across all cache configs

Any non-empty list means the resource is re-run per cache config when `--latencies-npy` is supplied.

### Naming convention (enforced by YAML)

| Layer | Convention | Example |
|---|---|---|
| YAML name / Python key / `.npy` stem | `snake_case` | `load_ls_pipes_lower` |
| C++ `Resource` enum | `UPPER_SNAKE` | `Resource::LOAD_LS_PIPES_LOWER` |
| C++ `ParamType` enum | `UPPER_SNAKE` | `ParamType::ROB_SIZE` |
| C++ function | `get_thr_<name>` | `get_thr_load_ls_pipes_lower` |
| Python `Config` field | param name | `rob_size` |

---

## How the C++ side is generated

Running `make` (or `python3 python/gen_registry.py`) reads `registry.yaml` and writes:

| File | Contents |
|---|---|
| `include/resources_gen.h` | `Resource` enum, one entry per resource in YAML order |
| `include/params_gen.h` | `ParamType` enum, `PARAM_RANGES` array, `get_param_range()`, `ParamSweep` |
| `include/resource_registry.h` | `RESOURCE_REGISTRY` — the vector that wires each resource to its `get_thr_*` function and param sweep |
| `include/models_decl_gen.h` | `get_thr_*` function declarations; `#include`d inside `models.h` |
| `python/config_gen.py` | `Config` dataclass with one field per param, defaults from registry |

The generated registry entry looks like:
```cpp
{
    Resource::ROB,
    "rob",
    true,   // enabled
    true,   // latency_dependent — re-run per cache config when --latencies-npy is used
    [](const auto& w, const auto& p) { return get_thr_rob(w, p[0]); },
    ParamSweep{ParamType::ROB_SIZE},
},
```

`get_throughput()` in `models.cpp` iterates `RESOURCE_REGISTRY`, skips disabled entries, and runs each function over all (param combo × window) pairs — in parallel via OpenMP.

---

## Workflow — step by step

### Prerequisites

```bash
# C++: g++ with C++20 and OpenMP
# Python: activate your venv
source ~/.venvs/berkourse/bin/activate
# PyYAML must be installed (uv pip install pyyaml)
```

All commands below are run from the **repo root** for Step 0 and from `anamol/` for Steps 1–6.

---

### Step 0 — Annotate the trace with cache latencies

Run from the **repo root** (not `anamol/`):

```bash
python annotate_trace.py anamol/traces/foo/trace.csv
```

This sweeps `(l1i_kb, l1d_kb, l2_kb, bp_id)` across 200 cache configurations, runs the full cache+branch-predictor simulation for each, and writes next to the input CSV:
- `anamol/traces/foo/trace_latencies.npy` — shape `(200, N_instrs, 2)` uint16: `[fetch_latency, exe_latency]` per instruction per config
- `anamol/traces/foo/trace_configs.json` — config index map for lookup

This step is slow (parallel via `ProcessPoolExecutor`). Re-run only when the trace changes or when the cache parameter grid changes.

---

### Step 1 — Edit `registry.yaml` (when needed)

This is the only file you edit to change resources or params. After editing, run `make` to regenerate headers and rebuild.

To **disable** a resource without deleting it:
```yaml
resources:
  - name: fetch_buffers
    enabled: false
```

To **change a sweep range**:
```yaml
params:
  - name: rob_size
    max: 2048   # extended range
```

---

### Step 2 — Build the C++ analytical model

```bash
make
```

This:
1. Runs `python3 python/gen_registry.py` if `registry.yaml` or `gen_registry.py` is newer than the generated headers
2. Compiles all sources (all objects depend on the generated headers, so a YAML change triggers a full rebuild)
3. Produces `build/main`, `build/test_params`, `build/test_models_omp`, `build/test_models_noomp`, `build/parser`

To build without OpenMP (serial):
```bash
make OMP=0
```

---

### Step 3 — Run the analytical model on a trace

If the trace lives in its own folder (the recommended layout — see [directory structure](#directory-structure)), pass `TRACE_DIR` and all three paths are derived automatically:

```bash
# Mode B (recommended) — folder layout:
make main.run TRACE_DIR=traces/foo WINDOW_SIZE=400

# Mode A — folder layout (no latency sweep):
make main.run TRACE_DIR=traces/foo WINDOW_SIZE=400 LATENCIES_NPY=
```

`TRACE_DIR=traces/foo` expands to `TRACE_CSV=traces/foo/trace.csv`, `LATENCIES_NPY=traces/foo/trace_latencies.npy`, `CONFIGS_JSON=traces/foo/trace_configs.json`, and output goes to `output/foo/`. Any variable can be overridden on the command line.

Alternatively, pass individual files:

#### Mode A — Single-latency (no cache sweep)

```bash
make main.run TRACE_CSV=traces/foo.csv WINDOW_SIZE=400
```

Trace CSV latency columns (`Fetch Latency`, `Execution Latency`) are optional — if absent they default to 0. Useful for quickly checking structural resource throughputs on a raw PIN trace. **Latency-dependent resources (ROB, LQ, SQ, icache_fills) will produce meaningless results with zero latencies** — use Mode B for those.

Writes to `output/foo/`:
- `thr_<resource>.npy` for every **enabled** resource — shape `(num_param_combos, param_cols + num_windows)`
- `rob_latency_overall_thr.npy`, `rob_latency_issue.npy`, `rob_latency_commit.npy`, `rob_latency_exec.npy`

#### Mode B — Per-cache-config (with latency sweep, recommended)

```bash
make main.run \
    TRACE_CSV=traces/foo.csv \
    WINDOW_SIZE=400 \
    LATENCIES_NPY=traces/foo/trace_latencies.npy
```

Requires the `.npy` produced by Step 0. Trace latency columns are ignored — latencies come from the `.npy` per config. Writes to `output/foo/`:
- **Latency-independent resources** (`latency_dependent: []`): written once in `output/foo/thr_<res>.npy`
- **Latency-dependent resources** (non-empty `latency_dependent`) + ROB analysis: written per config in `output/foo/config_0000/` … `config_0199/`

The `WINDOW_SIZE` passed here must match the value used in Python (Step 5).

---

### Step 4 — Build the throughput lookup table

Only needed after the analytical model output changes (new trace, changed param ranges, or toggled resources).

```bash
# Folder layout (recommended — derives output dir and configs-json automatically):
make lookup.run TRACE_DIR=traces/foo
```

Or with individual files:

#### Mode A (no cache sweep)
```bash
make lookup.run TRACE_CSV=traces/foo.csv
```

#### Mode B (with cache sweep — recommended)
```bash
make lookup.run \
    TRACE_CSV=traces/foo.csv \
    CONFIGS_JSON=traces/foo/trace_configs.json
```

`CONFIGS_JSON` / `trace_configs.json` maps cache parameter combinations to config indices and enables `lookup.get_config_features(config, cache_config)`. Without it the lookup is built in Mode A (no per-config indexing).

This reads all enabled `thr_*.npy` files and builds a lookup table keyed by:
- **Latency-independent resources**: `(param_values, ...)` → 101-dim feature vector
- **Latency-dependent resources** (when `CONFIGS_JSON` is given): `(cache_config_idx, *param_values)` → 101-dim feature vector

Each **101-dim feature vector** encodes the throughput distribution across all windows:
- 50 raw CDF percentiles (p1, p3, …, p99)
- 50 size-weighted CDF percentiles
- 1 mean value

To test the lookup table after building:
```bash
python3 python/build_throughput_lookup.py -i output/foo -o /tmp/test.pkl --test
```

---

### Step 5 — Generate training data

```bash
# N random configs (primary mode):
make training.run TRACE_DIR=traces/foo RANDOM_CONFIGS=1000 SEED=42

# Specific config (wrap the entire assignment in single quotes):
make training.run TRACE_DIR=traces/foo 'CONFIG_JSON={"rob_size":256,"load_queue_size":128}'
```

Output: `training/foo.pkl` (one row per config). `SEED` is optional. `RANDOM_CONFIGS` defaults to 1000. When `CONFIG_JSON` is set it takes precedence over `RANDOM_CONFIGS`.

#### Full pipeline (Steps 3–5 in one command)

```bash
make pipeline.run TRACE_DIR=traces/foo WINDOW_SIZE=400 RANDOM_CONFIGS=1000 SEED=42
```

Runs `main.run` → `lookup.run` → `training.run` in sequence. All the same `TRACE_DIR`, `WINDOW_SIZE`, `SEED`, and `CONFIG_JSON` variables apply.

#### Multi-trace: run and consolidate

```bash
# One full pipeline per trace:
make pipeline.run TRACE_DIR=traces/foo RANDOM_CONFIGS=1000 SEED=42
make pipeline.run TRACE_DIR=traces/bar RANDOM_CONFIGS=1000 SEED=42

# Merge all per-trace .pkl files into a single matrix:
make consolidate.run   # reads training/*.pkl → training/all_traces.pkl
```

`consolidate.run` reads every `*.pkl` in `training/` and stacks them row-wise into `training/all_traces.pkl`.

The output is a DataFrame with one row per config and the following column groups:

| Block | # Columns | Varies across configs? | Source |
|---|---|---|---|
| Per-resource throughput | `N_resources × 101` | Yes — selected from lookup by config | lookup table |
| Pipeline stall distributions | `4 × 101 = 404` | No — pure trace statistics | trace CSV |
| ROB latency distributions | `2334` | No — analytical model on full trace | `output/rob_latency_*.npy` |
| Config scalar features | `21 params + 4 one-hot = 25` | Yes | config object |

Current total with 11 enabled resources: **1111 + 404 + 2334 + 25 = 3874 columns** (varies with enabled resource count).

---

### Step 6 — Run tests

```bash
make test_params.run                   # validate param sweep counts

make test_models.run \                 # run all models on a trace, print throughputs
    TRACE_CSV=traces/collatz_trace_with_latency.csv \
    WINDOW_SIZE=400

make benchmark \                       # compare serial vs. OpenMP wall time
    TRACE_CSV=traces/collatz_trace_with_latency.csv \
    WINDOW_SIZE=400
```

---

## When to re-run each step

| What changed | Step 0 (annotate) | Step 2 (`make`) | Step 3 (model) | Step 4 (lookup) | Step 5 (training data) |
|---|:---:|:---:|:---:|:---:|:---:|
| `registry.yaml` — enable/disable resource | — | ✅ | ✅ | ✅ | ✅ |
| `registry.yaml` — change param range/default | — | ✅ | ✅ | ✅ | ✅ |
| C++ model logic (`src/models.cpp`) | — | ✅ | ✅ | ✅ | ✅ |
| New trace file | ✅ | — | ✅ | ✅ | ✅ |
| Cache parameter grid changed (`L1_KB`, `L2_KB`, `BP_IDS`) | ✅ | — | ✅ | ✅ | ✅ |
| Different configs for training only | — | — | — | — | ✅ |

---

## Adding, changing, and removing models

### Disable a resource (preferred)

```yaml
# registry.yaml
resources:
  - name: fetch_buffers
    enabled: false
```
Then `make`. The C++ function still exists but is never called.

### Add a new resource

1. **`registry.yaml`** — add param(s) if needed, add the resource entry:
   ```yaml
   params:
     - name: my_new_param
       min: 1
       max: 16
       step: base2
       default: 4
       enabled: true

   resources:
     - name: my_resource       # get_thr_my_resource will be called
       params: [my_new_param]
       enabled: true
   ```

2. **`src/models.cpp`** — implement the function:
   ```cpp
   double get_thr_my_resource(const vector<Instr>& window, uint16_t my_new_param) {
       // ...
       return (double)window.size() / total_cycles;
   }
   ```

3. `make` — generates all registry bindings and function declarations automatically.

### Change a model's logic

Edit the function body in `src/models.cpp`, then `make`. No YAML change needed unless the parameters change.

### Change a param's sweep range

Edit `min`/`max` in `registry.yaml`, then `make` → re-run the model → rebuild lookup table.

### Remove a resource (hard delete)

1. Delete from `registry.yaml`
2. Delete the function from `src/models.cpp`
3. Delete the param from `registry.yaml` if it was exclusive to this resource
4. `make`

### Add a new param (without a new resource)

1. Add an entry to `registry.yaml`
2. `make` — `config_gen.py` is regenerated automatically with the new field

---

## Python API (programmatic use)

```python
source ~/.venvs/berkourse/bin/activate
cd anamol/python/
```

```python
import sys
sys.path.insert(0, "python/")  # if not already on path

import models
import registry
from build_throughput_lookup import ThroughputLookupTable
from gen_training_data import (
    generate_training_sample,
    generate_training_matrix,
    build_training_data,
    sample_random_config,
    compute_pipeline_stall_features,
    compute_rob_latency_features,
)

# Load lookup table
lookup = ThroughputLookupTable.load("output/throughput_lookup.pkl")

# Single config
config = models.Config(rob_size=256, load_queue_size=128)
df = generate_training_sample(config, lookup, "traces/collatz_trace_with_latency.csv", window_size=400)

# Multiple configs
configs = [models.Config(rob_size=128), models.Config(rob_size=256), models.Config(rob_size=512)]
df = generate_training_matrix(configs, lookup, "traces/collatz_trace_with_latency.csv", window_size=400)

# Random configs
config = sample_random_config()   # draws each param independently from registry.yaml ranges

# Query lookup table directly (returns 101-dim array or None)
vec = lookup.query("rob", (128,))                        # latency-independent
vec = lookup.query("load_ls_pipes_lower", (2, 2))        # latency-independent

# For latency-dependent resources, include the cache config index as the first key element
# (only available when --configs-json was passed during build_throughput_lookup.py)
vec = lookup.query("rob", (cfg_idx, 128))                # latency-dependent

# Full feature vector for a config — requires cache_config dict for latency-dependent resources
cache_cfg = {"l1i_kb": 64, "l1d_kb": 64, "l2_kb": 1024, "bp_id": 1}
features = lookup.get_config_features(config, cache_config=cache_cfg)
features_df = lookup.get_config_features(config, cache_config=cache_cfg, as_dataframe=True)

# Just pipeline stall features (no lookup needed)
stall_df = compute_pipeline_stall_features("traces/collatz_trace_with_latency.csv", window_size=400)

# Just ROB latency features (for per-config mode, pass a config_NNNN/ subdirectory)
latency_df = compute_rob_latency_features("output/foo/config_0000/")
```

### Key Python modules

| Module | Purpose |
|---|---|
| `registry.py` | Reads `registry.yaml` at import time; provides `PARAMS`, `RESOURCES`, `PARAMS_BY_NAME`, `RESOURCES_BY_NAME`, `ENABLED_PARAMS`, `ENABLED_RESOURCES`, `RESOURCE_FILES`, etc. |
| `models.py` | `Resource` enum and re-exports from `registry` and `config_gen` |
| `config_gen.py` | **Generated** `Config` dataclass (defaults from registry) — do not edit manually |
| `utils.py` | `compute_cdf_features()`, `load_resource_file()`, `load_all_throughputs()`, `PERCENTILE_POINTS`, `FEATURES_PER_RESOURCE` |
| `build_throughput_lookup.py` | `ThroughputLookupTable` class: build, save, load, query, `get_config_features()` |
| `gen_training_data.py` | `generate_training_matrix()`, `sample_random_config()`, `compute_pipeline_stall_features()`, `compute_rob_latency_features()` |
| `gen_registry.py` | Code generator — run by `make`, not imported |

---

## Generated files (do not edit manually)

```
include/resources_gen.h
include/params_gen.h
include/resource_registry.h
include/models_decl_gen.h
python/config_gen.py
```

These are overwritten every time `make` runs if `registry.yaml` has changed. Add them to `.gitignore` if desired:
```
anamol/include/resources_gen.h
anamol/include/params_gen.h
anamol/include/resource_registry.h
anamol/include/models_decl_gen.h
anamol/python/config_gen.py
```

To regenerate manually without building:
```bash
python3 python/gen_registry.py --config registry.yaml --out-dir include
```

To delete generated files:
```bash
make clean-gen
```
