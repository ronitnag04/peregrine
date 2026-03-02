# Training Matrix Column Reference

> **Note:** The exact set of columns depends on which resources and parameters are enabled in [`registry.yaml`](../registry.yaml). The tables below reflect the current state of that file. Adding, removing, or toggling `enabled` on any resource or parameter will change the column count and names accordingly.

Both `gen_training_data.py` and `sweep_to_training.py` produce DataFrames with the same column layout (sections 1–4 below). The only difference is that `sweep_to_training.py` appends a `cpi` target column (section 5).

Percentile notation: **p1, p3, p5, …, p99** — 50 evenly-spaced odd integers from `np.linspace(1, 99, 50)`.
Each CDF feature block = 50 raw percentiles + 50 size-weighted percentiles + 1 mean = **101 columns**.

---

## Section 1 — Throughput Features (11 resources × 101 = 1 111 columns)

One CDF block per enabled resource, in registry order.
Source: `ThroughputLookupTable.get_config_features()` / `_compute_thr_features_from_output()`.

| Resource | Governing param | Latency-dependent |
|---|---|---|
| `rob` | `rob_size` | exe (cache config) |
| `load_queue` | `load_queue_size` | exe (cache config) |
| `store_queue` | `store_queue_size` | exe (cache config) |
| `alu_issue` | `alu_issue_width` | — |
| `alu_mult_div_issue` | `alu_mult_div_issue_width` | — |
| `fp_issue` | `fp_issue_width` | — |
| `fp_mult_div_issue` | `fp_mult_div_issue_width` | — |
| `ls_issue` | `ls_issue_width` | — |
| `load_ls_pipes_lower` | `num_ls_pipes`, `num_load_pipes` | — |
| `load_ls_pipes_upper` | `num_ls_pipes`, `num_load_pipes` | — |
| `icache_fills` | `max_icache_fills` | fetch (cache config) |

Column pattern for each resource `{res}`:
```
{res}_raw_p1, {res}_raw_p3, …, {res}_raw_p99          (50 cols)
{res}_weighted_p1, {res}_weighted_p3, …, {res}_weighted_p99  (50 cols)
{res}_mean                                              (1 col)
```

---

## Section 2 — Pipeline Stall Features (4 types × 101 = 404 columns)

Computed once per trace file, replicated across all configs.
Source: `compute_pipeline_stall_features()`.
Each value is a percentile/mean of the **per-window count** of that stall type.

| Prefix | Stall type | Trace column condition |
|---|---|---|
| `ISB` | Instruction sync barrier | `Instruction Sync == True` |
| `DIRECT_COND` | Direct conditional branch | `Branch Type == "direct_conditional"` |
| `DIRECT_UNCOND` | Direct unconditional branch | `Branch Type == "direct_unconditional"` |
| `INDIRECT` | Indirect branch | `Branch Type == "indirect"` |

Column pattern for each stall type `{type}`:
```
{type}_raw_p1, …, {type}_raw_p99          (50 cols)
{type}_weighted_p1, …, {type}_weighted_p99  (50 cols)
{type}_mean                                (1 col)
```

---

## Section 3 — ROB Latency Features (2 334 columns)

Loaded from pre-computed `.npy` files written by the C++ simulator.
Source: `compute_rob_latency_features()`.
The 11 ROB sizes are determined by the simulator sweep and encoded in `rob_latency_overall_thr.npy`.

### 3a. Overall throughput (11 columns)
```
rob{size}_overall_thr      — one column per ROB size
```

### 3b. Issue latency CDF (11 × 101 = 1 111 columns)
```
rob{size}_issue_raw_p1, …, rob{size}_issue_raw_p99
rob{size}_issue_weighted_p1, …, rob{size}_issue_weighted_p99
rob{size}_issue_mean
```

### 3c. Commit latency CDF (11 × 101 = 1 111 columns)
```
rob{size}_commit_raw_p1, …, rob{size}_commit_raw_p99
rob{size}_commit_weighted_p1, …, rob{size}_commit_weighted_p99
rob{size}_commit_mean
```

### 3d. Exec latency CDF — averaged across all ROB sizes (101 columns)
```
exec_raw_p1, …, exec_raw_p99
exec_weighted_p1, …, exec_weighted_p99
exec_mean
```

**Total section 3: 11 + 1 111 + 1 111 + 101 = 2 334 columns**

---

## Section 4 — Config Scalar Features (25 columns)

Source: `get_config_scalar_features()`.

### 4a. Scalar microarchitecture parameters (21 columns)

One column per enabled param, named exactly as the `Config` field:

| Column | Range | Description |
|---|---|---|
| `rob_size` | 1–1024 | Reorder buffer entries |
| `commit_width` | 1–12 | Commit bandwidth (instructions/cycle) |
| `load_queue_size` | 1–256 | Load queue entries |
| `store_queue_size` | 1–256 | Store queue entries |
| `alu_issue_width` | 1–8 | Integer ALU issue ports |
| `alu_mult_div_issue_width` | 1–8 | Integer multiply/divide issue ports |
| `fp_issue_width` | 1–8 | FP ALU issue ports |
| `fp_mult_div_issue_width` | 1–8 | FP multiply/divide issue ports |
| `ls_issue_width` | 1–16 | Load/store issue width (rdwr + read ports) |
| `num_ls_pipes` | 1–8 | Read-write (store) LS pipes |
| `num_load_pipes` | 1–8 | Read-only (load) LS pipes |
| `fetch_width` | 1–12 | Fetch bandwidth |
| `decode_width` | 1–12 | Decode bandwidth |
| `rename_width` | 1–12 | Rename bandwidth |
| `max_icache_fills` | 1–32 | Max outstanding I-cache fills |
| `branch_predictor` | 0 or 1 | 0 = local, 1 = TAGE (raw integer) |
| `misprediction_percent` | 0.0–1.0 | Branch misprediction rate (float) |
| `l1d_cache_kb` | 16–256 | L1D cache size (KB, powers of 2) |
| `l1i_cache_kb` | 16–256 | L1I cache size (KB, powers of 2) |
| `l2_cache_kb` | 512–4096 | L2 cache size (KB, powers of 2) |
| `l1d_stride_prefetch` | 0 or 4 | Stride prefetcher degree (0 = off) |

### 4b. One-hot encoded categorical features (4 columns)

| Column | Value |
|---|---|
| `bp_is_simple` | 1 if `branch_predictor == 0` (local), else 0 |
| `bp_is_tage` | 1 if `branch_predictor == 1` (TAGE), else 0 |
| `prefetcher_off` | 1 if `l1d_stride_prefetch == 0`, else 0 |
| `prefetcher_on` | 1 if `l1d_stride_prefetch != 0`, else 0 |

---

## Section 5 — Target (sweep_to_training.py only, 1 column)

| Column | Description |
|---|---|
| `cpi` | Cycles per instruction from gem5 simulation (regression target) |

---

## Summary

| Section | Source | Columns |
|---|---|---|
| 1. Throughput CDF | Lookup table / anamol `.npy` | 11 × 101 = **1 111** |
| 2. Pipeline stall CDF | Trace CSV | 4 × 101 = **404** |
| 3. ROB latency CDF | Simulator `.npy` | **2 334** |
| 4. Config scalars + one-hot | Config object | **25** |
| **Total (features)** | | **3 874** |
| 5. Target (`cpi`) | gem5 sweep CSV | **1** (sweep only) |

The column order within each section is determined by:
- Section 1: registry.yaml resource order
- Section 2: `ISB → DIRECT_COND → DIRECT_UNCOND → INDIRECT`
- Section 3: ROB size ascending (as stored in `.npy`)
- Section 4: registry.yaml param order, then one-hot columns appended
