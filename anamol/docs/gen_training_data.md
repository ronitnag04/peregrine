# Analytical Models → Lookup Table → Training Data

All commands run from `anamol/`.

---

## Normal workflow

All `make` commands run from `anamol/`.
The `annotate_trace.py` script runs from the **repo root**.

### 0. Annotate the trace with cache latencies (once per trace)

```bash
# Run from repo root:
python annotate_trace.py anamol/traces/foo/trace.csv
# → anamol/traces/foo/trace_latencies.npy   (200 configs × N_instrs × 2 uint16)
# → anamol/traces/foo/trace_configs.json    (config index map)
```

Re-run only when the trace changes or the cache grid (`L1_KB`, `L2_KB`, `BP_IDS`) changes.

### 1. Edit resources/params (one file only)

```bash
# anamol/registry.yaml
# - Add/remove/enable/disable resources and params here
# - Set latency_dependent: [exe] / [fetch] / [exe, fetch] / [] for each resource
# e.g. set enabled: false to skip a resource
```

### 2. Regenerate C++ headers + rebuild

```bash
cd anamol/

# Headers are auto-regenerated whenever registry.yaml changes:
make

# Or manually regenerate without full build:
python3 python/gen_registry.py --config registry.yaml
```

### 3. Run the C++ analytical model on a trace

**With cache latency sweep (recommended):**
```bash
make main.run TRACE_DIR=traces/foo WINDOW_SIZE=400
```

Writes to `output/foo/`:
- Latency-independent resources: `thr_<res>.npy` written once
- Latency-dependent resources + ROB analysis: `config_0000/thr_<res>.npy` … `config_0199/`

**Without cache latency sweep (single-latency mode):**
```bash
make main.run TRACE_DIR=traces/foo WINDOW_SIZE=400 LATENCIES_NPY=
```

Writes `output/foo/thr_<resource>.npy` for every enabled resource, plus `output/foo/rob_latency_*.npy`.

### 4. Build the throughput lookup table

Only re-run when analytical model output changes (new trace, changed param ranges, or toggled resources).

```bash
source ~/.venvs/berkourse/bin/activate

make lookup.run TRACE_DIR=traces/foo
```

### 5. Generate training data

```bash
# N random configs:
make training.run TRACE_DIR=traces/foo RANDOM_CONFIGS=1000 SEED=42

# Specific config (wrap the entire assignment in single quotes):
make training.run TRACE_DIR=traces/foo 'CONFIG_JSON={"rob_size":256,"load_queue_size":128}'
```

Output: `training/foo.pkl`. `SEED` is optional. When `CONFIG_JSON` is set it takes precedence over `RANDOM_CONFIGS`.

### Full pipeline (steps 3–5 in one command)

```bash
make pipeline.run TRACE_DIR=traces/foo WINDOW_SIZE=400 RANDOM_CONFIGS=1000 SEED=42
```

Runs `main.run` → `lookup.run` → `training.run` in sequence.

### Multi-trace: run and consolidate

```bash
# One full pipeline per trace:
make pipeline.run TRACE_DIR=traces/foo RANDOM_CONFIGS=1000 SEED=42
make pipeline.run TRACE_DIR=traces/bar RANDOM_CONFIGS=1000 SEED=42

# Merge all per-trace .pkl files into a single matrix:
make consolidate.run   # reads training/*.pkl → training/all_traces.pkl
```

---

## When to re-run each step

| What changed | Step 0 (annotate) | `make` | Step 3 (sim) | Step 4 (lookup) | Step 5 (training) |
|---|:---:|:---:|:---:|:---:|:---:|
| `registry.yaml` — add/remove/change param or resource | — | ✅ | ✅ | ✅ | ✅ |
| `registry.yaml` — just `enabled: false` a resource | — | ✅ | ✅ | ✅ | ✅ |
| C++ model code (`src/*.cpp`) | — | ✅ | ✅ | ✅ | ✅ |
| New trace file | ✅ | — | ✅ | ✅ | ✅ |
| Cache grid changed (`L1_KB`, `L2_KB`, `BP_IDS`) | ✅ | — | ✅ | ✅ | ✅ |
| Different configs for training data only | — | — | — | — | ✅ |

---

## Adding / removing / changing a model

A "model" = one `get_thr_<name>` C++ function + its entry in `registry.yaml`.

### Disable a resource (preferred)

```yaml
# registry.yaml
resources:
  - name: fetch_buffers
    enabled: false
```

Then `make`. The function still exists in `models.cpp` but is never called. Its `.npy` file is not written.

### Add a new resource

**1. Add param(s) to `registry.yaml`** (if the resource needs a new knob):

```yaml
params:
  - name: my_new_param
    min: 1
    max: 16
    step: base2
    default: 4
    enabled: true
```

**2. Add the resource to `registry.yaml`:**

```yaml
resources:
  - name: my_resource        # C++ function must be named get_thr_my_resource
    params: [my_new_param]   # 1 or 2 params only
    enabled: true
```

**3. Implement the C++ function in `src/models.cpp`:**

```cpp
double get_thr_my_resource(const vector<Instr>& window, uint16_t my_new_param) {
    // ... model logic ...
    return (double)window.size() / total_cycles;
}
```

**4. Set `latency_dependent` in `registry.yaml`:**

```yaml
resources:
  - name: my_resource
    params: [my_new_param]
    enabled: true
    latency_dependent: []      # [exe] if calls resp_cycle(), [fetch] if uses fetch_latency, [] otherwise
```

**5. `make`** — generates the declaration in `models_decl_gen.h`, the registry binding in `resource_registry.h`, and the `Config` field in `config_gen.py` automatically.

### Change a model's logic

Edit the function body in `src/models.cpp`, then `make`. No YAML changes needed unless the parameters themselves change.

### Change a param's sweep range

Edit `min`/`max` in `registry.yaml`, then `make` → re-run the model → rebuild lookup.

### Remove a resource (hard delete)

1. Delete from `registry.yaml`
2. Delete the function from `src/models.cpp` and its declaration from `include/models.h`
3. Delete the `Config` field from `python/models.py` (and the param entry from `registry.yaml` if it was exclusive to this resource)
4. `make`

---

## Calling from Python

Run from `anamol/python/` or add `python/` to `sys.path` first.

### Single config

```python
from gen_training_data import generate_training_sample
from build_throughput_lookup import ThroughputLookupTable
import models

lookup = ThroughputLookupTable.load("output/throughput_lookup.pkl")
config = models.Config(rob_size=256, load_queue_size=128)

training_data = generate_training_sample(
    config,
    lookup,
    "traces/collatz_trace_with_latency.csv",
    window_size=400,
)
```

### Multiple configs

```python
from gen_training_data import generate_training_matrix
from build_throughput_lookup import ThroughputLookupTable
import models

lookup = ThroughputLookupTable.load("output/throughput_lookup.pkl")
configs = [
    models.Config(rob_size=128),
    models.Config(rob_size=256, load_queue_size=128),
]

training_data = generate_training_matrix(
    configs,
    lookup,
    "traces/collatz_trace_with_latency.csv",
    window_size=400,
)
```

### Random configs

```python
from gen_training_data import sample_random_config, generate_training_matrix
from build_throughput_lookup import ThroughputLookupTable

lookup = ThroughputLookupTable.load("output/throughput_lookup.pkl")
configs = [sample_random_config() for _ in range(1000)]

training_data = generate_training_matrix(
    configs,
    lookup,
    "traces/collatz_trace_with_latency.csv",
    window_size=400,
)
```

### Convenience wrapper (loads lookup automatically)

```python
from gen_training_data import build_training_data
import models

training_data = build_training_data(
    lookup_path="output/throughput_lookup.pkl",
    trace_path="traces/collatz_trace_with_latency.csv",
    configs=[models.Config(rob_size=256, load_queue_size=128)],
    window_size=400,
    output_path="training_data.pkl",  # optional — save to disk
)
```
