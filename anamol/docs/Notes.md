# Analytical Modeling

## 3.2.1 Per Resource Throughput Dist
**11 * 101**

1. ROB
2. Load queue
3. Store queue
4. ALU issue width
5. Floating-point issue width
6. Load-store issue width
7. Load/Load-Store Pipes lower
8. Load/Load-Store Pipes upper
9.  I-cache fills
10. Fetch Buffers
~~11. ??? TLB~~

## 3.2.2

### Pipeline Stalls
**4 * 101**

Distributions:
1. ISB
2. DIRECT_COND Branches
3. DIRECT_UNCOND Branches
4. INDIRECT Branches

**1**
Scalar:
1. Overall Branch-Misprediction Rate

**11 * 1**
For ROB size in {1,2,4,8,...,1024} (n=11)
1. Overall ROB throughput (not windowed)

### Latency Distributions

**11 * 2 * 101**
For ROB size in {1,2,4,8,...,1024} (n=11):
1. Issue Latency $s_i - a_i$ (n=101)
2. Commit Latency $c_i - f_i$ (n=101)

**101**
Same for all ROB:
1. Exec Latency $f_i - s_i$ (n=101)

# TODO
1. trace conversion confirmation (category)
2. icache simulation confirmation


### One Training Row = (Trace Region R, Architecture P)

| Block | Feature Group | Depends on | # Columns | Description |
|------:|--------------|------------|-----------|-------------|
| 1 | Per-resource throughput | Trace R + Arch P | **1111** | 11 resources x 101-dim CDF encodings. Selected from lookup tables using config P. |
| 2 | Pipeline stall distributions | Trace R only | **416** | Program behavior summaries (ISBs, branch types, mispredict rate, ROB sensitivity). |
| 3 | Latency distributions | Trace R only | **2323** | Execution, issue, and commit latency CDFs across all ROB sizes. |
| 4 | Architecture encoding | Arch P only | **23** | Numeric params + one-hot branch predictor + prefetcher. |
|   | **TOTAL** | — | **3873** | One complete feature vector. |

### Detailed Feature Composition (One Row)

| Block | Sub-feature | # Columns | Notes |
|------:|-------------|-----------|------|
| 1 | ROB throughput CDF | 101 | Selected by ROB size |
|   | Load Queue throughput CDF | 101 | Selected by LQ size |
|   | Store Queue throughput CDF | 101 | Selected by SQ size |
|   | ALU issue throughput CDF | 101 | Selected by width |
|   | FP issue throughput CDF | 101 | Selected by width |
|   | LS issue throughput CDF | 101 | Selected by width |
|   | Load/LS pipes (lower) CDF | 101 | Selected by (LP, LSP) |
|   | Load/LS pipes (upper) CDF | 101 | Selected by (LP, LSP) |
|   | I-cache fills CDF | 101 | Selected by fill limit |
|   | Fetch buffers CDF | 101 | Selected by buffer count |
|   | (11 total resources) | **1111** | |
| 2 | ISB count CDF | 101 | Per-window counts |
|   | Direct conditional branch count CDF | 101 | Per-window counts |
|   | Direct unconditional branch count CDF | 101 | Per-window counts |
|   | Indirect branch count CDF | 101 | Per-window counts |
|   | Branch mispredict rate | 1 | Scalar |
|   | ROB throughput vs ROB size | 11 | Scalars (ROB = 1..1024) |
|   | **Pipeline stalls total** | **416** | |
| 3 | Execution latency CDF | 101 | Same for all ROB sizes |
|   | Issue latency CDFs | 11 x 101 | One per ROB size |
|   | Commit latency CDFs | 11 x 101 | One per ROB size |
|   | **Latency total** | **2323** | |
| 4 | Architecture encoding | 23 | Numeric + one-hot |

### Column Variability Across Rows Sharing the Same Trace Region R

| Feature Block | # Columns | Varies Across Rows? | Reason |
|--------------|-----------|--------------------|-------|
| Per-resource throughput | 1111 | ✅ YES | Selected from lookup using different architecture configs. |
| Pipeline stall distributions | 416 | ❌ NO | Pure program behavior from the trace. |
| Latency distributions | 2323 | ❌ NO | Program + analytical ROB model, included in full every time. |
| Architecture encoding | 23 | ✅ YES | Explicit description of config P. |
| CPI (label) | 1 | ✅ YES | Depends on interaction of program and architecture. |

### Full Training Matrix Structure

| Dimension | Meaning |
|----------|--------|
| Rows | (Trace region R, Architecture P) pairs |
| Columns | Fixed at 3873 |
| X shape | (N, 3873) |
| y shape | (N,) CPI labels |

Each trace region R contributes many rows:
- Same pipeline stalls
- Same latency distributions
- Different throughput selections
- Different architecture encodings
