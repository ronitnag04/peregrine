# Run Steps: Analytical Models -> Lookup Table -> Training Data

Run these commands from:

```bash
cd ./anamol
```

## 1) Run the analytical models

```bash
make main.run OMP=0 TRACE_CSV=traces/collatz_trace_with_latency.csv WINDOW_SIZE=400
```

This generates throughput outputs in `output/`.

## 2) Build the throughput lookup table

```bash
python python/build_throughput_lookup.py -i output/ -o output/collatz_w_lat.pkl
```

## 3) Generate training data

```bash
python python/gen_training_data.py \
  --lookup output/collatz_w_lat.pkl \
  --config-json '{"rob_size": 256, "load_queue_size": 128}' \
  --trace traces/collatz_trace_with_latency.csv \
  --window-size 400 \
  -o training_data.pkl
```

## Calling from Python

Use one of these two setups:

- Run from `anamol/python` and import modules directly (`from gen_training_data import ...`).
- Or run from `anamol/` and add `python/` to `sys.path` before imports.

### A) Single config (`generate_training_sample`)

```python
from gen_training_data import generate_training_sample
from build_throughput_lookup import ThroughputLookupTable
import models

lookup = ThroughputLookupTable.load("output/collatz_w_lat.pkl")
config = models.Config(rob_size=256, load_queue_size=128)

training_data = generate_training_sample(
    config,
    lookup,
    "traces/collatz_trace_with_latency.csv",
    window_size=400,
)
```

### B) Multiple configs (`generate_training_matrix`)

```python
from gen_training_data import generate_training_matrix
from build_throughput_lookup import ThroughputLookupTable
import models

lookup = ThroughputLookupTable.load("output/collatz_w_lat.pkl")
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

### C) Convenience wrapper (`build_training_data`)

```python
from gen_training_data import build_training_data
import models

training_data = build_training_data(
    lookup_path="output/collatz_w_lat.pkl",
    trace_path="traces/collatz_trace_with_latency.csv",
    configs=[models.Config(rob_size=256, load_queue_size=128)],
    window_size=400,
)
```
