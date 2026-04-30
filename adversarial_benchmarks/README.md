# Adversarial benchmarks for Ronamol (analytical + ML pipeline)

These C benchmarks are designed to expose blind spots in the feature
set computed by `peregrine/ronamol/python/simple_model.py`. Each
benchmark is constructed so that its analytical feature vector looks
"innocuous" — or is a near-duplicate of a very different workload —
while the real CPI is dominated by micro-architectural effects the
feature extractor cannot see.

The goal is *not* to make fast or slow code but to create labelled
points the analytical pipeline is structurally incapable of
distinguishing from unrelated workloads.

## Scaling note (SPEC-style)

SPEC benchmarks run for 100Ms–10Bs of dynamic instructions, and the
ronamol training pipeline samples 100k-instruction regions from those
traces. Each benchmark below is sized to issue **multiple billions**
of dynamic instructions by default, so a 100k-insn sampling window
never sees the startup / teardown transients. Every knob is
overridable three ways:

1. `-D<KNOB>=<value>` at build time (see the `Makefile`)
2. A process-specific env var (e.g. `PTRCHASE_HOPS`, `ICACHE_BLAST_ITERS`)
3. Editing the default in the source (all at the top of each file)

Default dynamic instruction counts (calibrated against `perf stat`):

| Benchmark                        | default dyn insts |
|----------------------------------|-------------------|
| `pow2_stride_thrash` / `_benign` | ~10B              |
| `icache_blast`                   | ~10B              |
| `ptrchase_rand`                  | ~1B (latency-bound; kept small) |
| `serial_mul_chain`               | ~10B              |
| `many_pages_streaming`           | ~10B              |
| `adversarial_branches`           | ~10B              |
| `stlf_misalign`                  | ~10B              |

## Feature blind spots targeted

| Blind spot in `simple_model.py` | Benchmark |
|---|---|
| `load_stride_regularity` rewards *any* constant stride, ignoring set-conflict geometry | `pow2_stride_thrash.c` (paired with `pow2_stride_benign`) |
| No feature measures static code footprint / I-cache pressure | `icache_blast.c` |
| Pointer-chasing vs. streaming over the same number of pages is indistinguishable (no MLP feature) | `ptrchase_rand.c` vs `many_pages_streaming.c` |
| A tight producer→consumer chain (`max_dist == 1`) registers as *non-critical* (`crit_path_density_10 == 0`), masking a fully serialized pipeline | `serial_mul_chain.c` |
| `unique_load_pages` punishes large footprints, even when access is streaming and prefetch-friendly | `many_pages_streaming.c` |
| Branch types are binned but mis-prediction pattern shape (e.g., coprime-period interleave that defeats TAGE / local history) is not | `adversarial_branches.c` |
| `frac_memory_ordering_hazards` fires on `len(mem_dependent_ips) > 1`, ignoring store-to-load-forwarding alignment pathology with a single producer | `stlf_misalign.c` |

## Why each one fools the pipeline

### 1. `pow2_stride_thrash.c`  (pair: `pow2_stride_benign`)
Single load IP with a `(1 << STRIDE_LOG2)`-byte stride.
`simple_model.py:250` computes `best / total_deltas` per IP and
averages — this benchmark produces `load_stride_regularity = 1.0`,
the **max possible score**. The `_thrash` variant (`STRIDE_LOG2=15`,
32 KiB) lands every access in the same L1D set and triggers
associativity conflicts. The `_benign` variant (`STRIDE_LOG2=6`,
64 B = sequential) is the cache-friendly sibling. Both have **the
same** analytical feature vector but differ in CPI by an order of
magnitude.

### 2. `icache_blast.c`
Dispatches pseudo-randomly through hundreds of distinct tiny
callees. Default code size with `N_FUNCS=512` and `PAD_OPS=12` is
comfortably past a typical 32 KiB L1I. Dynamic instruction mix
looks boring (int ALU + one indirect branch + return per iter);
the feature vector resembles a tiny loop. `simple_model.py` has
*no* feature representing code footprint — the model will predict
low CPI, but real CPI is dominated by I-cache / BTB / I-TLB
misses. To dial footprint up or down, edit `N_FUNCS` in
`gen_icache_blast.py`.

### 3. `ptrchase_rand.c`  (pair: `many_pages_streaming.c`)
Serial pointer chase over a 256 MiB shuffled cycle. The analytical
model cannot separate "1 outstanding miss at a time" (MLP = 1)
from "64 outstanding misses at a time" (MLP high), because
`frac_mem_dependent` collapses both to ~1.0. Paired with
`many_pages_streaming.c` sized to the same `unique_load_pages`,
the two benchmarks look much more similar than their CPI warrants.

### 4. `serial_mul_chain.c`
A single long chain `x = x * k + c`. Every consumer's producer is
the previous instruction, so `mean_reg_dep_distance = 1`,
`p95_reg_dep_distance = 1`, `crit_path_density_10 = 0`. The
feature set says "lots of short deps, no critical path"; the
workload is as serial as a program can be.

### 5. `many_pages_streaming.c`
Sequential scan of an N-page buffer sized to match the pointer-chase
companion. `simple_model.py` treats `unique_load_pages` as a coarse
working-set measure; streaming collapses most misses via HW
prefetch. The pipeline, lacking any feature for *access order*,
will apply the same high-pages penalty it does to the pointer chase.

### 6. `adversarial_branches.c`
Dense `direct_conditional` branches with three coprime periods
(5, 7, 11) plus a long-period data-derived branch. Branch-type
fractions are identical to a well-behaved loop-bounded program,
but predictor accuracy collapses. Where only `simple_model.py`
features are used (the BP-rate sidecar may mitigate this in the
full pipeline, but several downstream paths use features alone),
the misprediction pattern is invisible.

### 7. `stlf_misalign.c`
1-byte store to offset 0, followed by an 8-byte load of the same
qword. `len(mem_dependent_ips) == 1`, so
`frac_memory_ordering_hazards = 0`. Every iteration nevertheless
incurs a store-to-load-forwarding stall. The model sees "no
ordering hazard" but reality pays a ~10-cycle bubble per load.

## Build

```
cd /home/ubuntu/peregrine/benchmarks
make            # builds all 8 binaries under build/bin/
make clean      # wipes the entire build/ tree
```

All generated artifacts live under `build/`:

```
build/
├── bin/        # executables
├── obj/        # .o files
└── gen/        # generated headers (icache_blast_gen.h)
```

Source files stay clean. `build/` is `.gitignore`d.

To override the dynamic instruction count for a specific benchmark
without rebuilding, set the corresponding env var:

```
ICACHE_BLAST_ITERS=50000000  build/bin/icache_blast
PTRCHASE_HOPS=1000000000     build/bin/ptrchase_rand
SERIAL_CHAIN_ITERS=200000000 build/bin/serial_mul_chain
MPS_PASSES=4                 build/bin/many_pages_streaming
```

Tracing with `evantrace` and running through `gen_features.py`,
`gen_cache_latency.py`, and `gen_bp_rate.py` is the next step —
not wired in here, since the tracer entry point lives elsewhere
in the peregrine tree (see `peregrine/sweep_traces.sh`).
