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
├── traces/                  # Input dynamic instruction traces (.csv)
└── output/                  # C++ simulator output: thr_*.npy, rob_latency_*.npy
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

  - name: load_ls_pipes_lower
    params: [num_ls_pipes, num_load_pipes]   # 2-param resource
    enabled: true

  - name: fetch_buffers
    params: [num_fetch_buffers]
    enabled: false                   # skipped by C++ and Python
```

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

All commands below are run from `anamol/`.

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

### Step 2 — Build the C++ simulator

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

### Step 3 — Run the simulator on a trace

```bash
make main.run TRACE_CSV=traces/collatz_trace_with_latency.csv WINDOW_SIZE=400

# Or directly:
./build/main -t traces/collatz_trace_with_latency.csv -w 400
```

This writes to `output/`:
- `thr_<resource>.npy` for every **enabled** resource — shape `(num_param_combos, param_cols + num_windows)`
- `rob_latency_overall_thr.npy` — shape `(11, 2)`: ROB size vs. overall throughput
- `rob_latency_issue.npy`, `rob_latency_commit.npy`, `rob_latency_exec.npy` — shape `(11, num_instructions)`: per-instruction latencies for 11 ROB sizes

The `window_size` passed here must match the value used in Python (Step 5).

---

### Step 4 — Build the throughput lookup table

Only needed after the simulator output changes (new trace, changed param ranges, or toggled resources).

```bash
python3 python/build_throughput_lookup.py \
    -i output/ \
    -o output/throughput_lookup.pkl
```

This reads all enabled `thr_*.npy` files and builds a lookup table:
```
{resource_name: {(param_values,...): 101-dim feature vector}}
```

Each **101-dim feature vector** per (resource, param combo) encodes the throughput distribution across all windows:
- 50 raw CDF percentiles (p1, p3, …, p99)
- 50 size-weighted CDF percentiles
- 1 mean value

To test the lookup table after building:
```bash
python3 python/build_throughput_lookup.py -i output/ -o /tmp/test.pkl --test
```

---

### Step 5 — Generate training data

```bash
# Single config (with specific overrides from defaults):
python3 python/gen_training_data.py \
    --lookup output/throughput_lookup.pkl \
    --trace traces/collatz_trace_with_latency.csv \
    --window-size 400 \
    --config-json '{"rob_size": 256, "load_queue_size": 128}' \
    -o training_data.pkl

# N random configs:
python3 python/gen_training_data.py \
    --lookup output/throughput_lookup.pkl \
    --trace traces/collatz_trace_with_latency.csv \
    --window-size 400 \
    --random-configs 1000 \
    --seed 42 \
    -o training_data.pkl   # or .csv
```

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

| What changed | Step 2 (`make`) | Step 3 (simulator) | Step 4 (lookup) | Step 5 (training data) |
|---|:---:|:---:|:---:|:---:|
| `registry.yaml` — enable/disable resource | ✅ | ✅ | ✅ | ✅ |
| `registry.yaml` — change param range/default | ✅ | ✅ | ✅ | ✅ |
| C++ model logic (`src/models.cpp`) | ✅ | ✅ | ✅ | ✅ |
| New trace file | — | ✅ | ✅ | ✅ |
| Different configs for training only | — | — | — | ✅ |

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

Edit `min`/`max` in `registry.yaml`, then `make` → re-run simulator → rebuild lookup table.

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
vec = lookup.query("rob", (128,))
vec = lookup.query("load_ls_pipes_lower", (2, 2))

# Full feature vector for a config (returns numpy array of shape (N_resources * 101,))
features = lookup.get_config_features(config)
features_df = lookup.get_config_features(config, as_dataframe=True)

# Just pipeline stall features (no lookup needed)
stall_df = compute_pipeline_stall_features("traces/collatz_trace_with_latency.csv", window_size=400)

# Just ROB latency features
latency_df = compute_rob_latency_features("output/")
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
