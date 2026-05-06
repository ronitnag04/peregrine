# Analytical & gem5 Model Specifications

## Table of Contents

**Out-of-order Core Parameters:**
1. [Re-order Buffer](#re-order-buffer)
2. [Load Queue](#load-queue)
3. [Store Queue](#store-queue)
4. [Commit Width](#commit-width)
5. [Writeback Width](#writeback-width)
6. [Fetch Width](#fetch-width)
7. [Decode Width](#decode-width)
8. [Rename Width](#rename-width)
9. [ALU Issue Width](#alu-issue-width)
10. [Floating-Point Issue Width](#floating-point-issue-width)
11. [Load-Store Issue Width](#load-store-issue-width)
12. [Load and Load-Store Pipes](#load-and-load-store-pipes)
13. [Fetch Buffers](#fetch-buffers)
14. [Maximum I-Cache Fills](#maximum-i-cache-fills)
15. [Cache Configuration](#cache-configuration)
16. [Branch Predictor](#branch-predictor)

**Other:**
[How to export this file to a PDF](#how-to-export-this-file-to-a-pdf)

# Re-order Buffer
## Description
The re-order buffer (ROB) stores in-flight instructions and enables out-of-order/speculative execution by
forcing instructions to be committed (i.e. result written to register or memory) in-order. The stored
instructions are used to track data dependencies, and the head points to the oldest instruction yet to be
committed.

## Model Overview

The throughput bound imposed by the ROB is determined by its parametrizable size and the latencies of the
instructions in the given window. Since the head of the ROB only advances once the oldest instruction is
committed, and it can only hold so many instructions at once, the pipeline needs to be stalled when
instructions with long latencies are encountered.

For each instruction $i$, the model tracks four cycle timestamps:

| Variable | Meaning |
|---|---|
| $a_i$ | Arrival cycle — when instruction $i$ enters the ROB |
| $s_i$ | Start cycle — when instruction $i$ begins execution |
| $f_i$ | Finish cycle — when instruction $i$ completes execution |
| $c_i$ | Commit cycle — when instruction $i$ retires from the ROB |

These are computed via the following equations:

$$a_i = c_{i - \text{ROB}}$$
$$s_i = \max\!\left(a_i,\ \max_{d \in \text{Dep}(i)} f_d\right)$$
$$f_i = \text{RespCycle}(s_i, \text{instr}_i)$$
$$c_i = \max(f_i,\ c_{i-1})$$

with the convention that $c_i = 0$ for $i \leq 0$.

### What each equation captures

**Equation 1 — ROB size constraint.** An instruction can only enter the ROB once the instruction that is `ROB`-slots ahead of it has committed, thus modeling the chosen ROB size.

**Equation 2 — Instruction dependency constraint.** An instruction cannot begin executing until all of its register and memory dependencies (the set $\text{Dep}(i)$, extracted during trace analysis) have finished executing. The start cycle is therefore the maximum of the instruction's arrival time and the finish times of all its dependencies.

**Equation 3 — Execution finish time.** The finish cycle is computed using `RespCycle`, which is a small state machine that models memory behaviour (will add a section for this algorithm).

**Equation 4 — In-order commit constraint.** Instructions must commit in program order, so the commit cycle of instruction $i$ is the maximum of it's own finish cycle and the commit cycle of the previous instruction $i-1$.

### Computing the Throughput Bound

Once the commit cycles are computed, Concorde divides the program region into consecutive windows of $k$ instructions (the paper uses $k = 400$, roughly the size of a ROB). The throughput for the $j$-th window is:

$$\text{thr}_j^{\text{ROB}} = \frac{k}{c_{kj} - c_{k(j-1)}}$$

This gives the instructions-per-cycle (IPC) achievable if the ROB were the *only* bottleneck.

## gem5 Representation

We configure the ROB size of our core in gem5 by setting the already exposed `numROBEntries` attribute of the
`X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Load Queue
## Description
The load queue stores in-flight load instructions until they are committed.

## Model Overview

Since the load queue has a finite, parametrizable size, it sets a bound on the number of in-flight loads at any
given time. 

The same four per-instruction timestamps used in the ROB model are tracked, but applied only to load instructions:

| Variable | Meaning |
|---|---|
| $a_i$ | Arrival cycle — when load $i$ obtains a slot in the Load Queue |
| $s_i$ | Start cycle — when load $i$ begins executing |
| $f_i$ | Finish cycle — when load $i$ completes (data is available) |
| $c_i$ | Commit cycle — when load $i$ retires and frees its queue slot |

The equations mirror the ROB model:

$$a_i = c_{i - \text{LQ}}$$
$$s_i = a_i$$
$$f_i = \text{RespCycle}(s_i, \text{instr}_i)$$
$$c_i = \max(f_i,\ c_{i-1})$$

Note however that unlike the ROB model, dependency constraints do not play a role because a load instruction is
eligible to start as soon as it obtains a slot in the queue.

And the throughput for the $j$-th window of $k$ instructions is computed the same way:

$$\text{thr}_j^{\text{LQ}} = \frac{k}{c_{kj} - c_{k(j-1)}}$$

## gem5 Representation

We configure the load queue size of our core in gem5 by setting the already exposed `LQEntries` attribute of the
`X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Store Queue
## Description
The store queue stores in-flight store instructions until they are committed. 

## Model Overview

Since the store queue has a finite, parametrizable size, it sets a bound on the number of in-flight stores at any
given time.

The same four per-instruction timestamps used in the ROB model are tracked, but applied only to store
instructions:

| Variable | Meaning |
|---|---|
| $a_i$ | Arrival cycle — when store $i$ obtains a slot in the Store Queue |
| $s_i$ | Start cycle — when store $i$ begins executing |
| $f_i$ | Finish cycle — when store $i$ completes (data is available) |
| $c_i$ | Commit cycle — when store $i$ retires and frees its queue slot |

The equations mirror the ROB model:

$$a_i = c_{i - \text{SQ}}$$
$$s_i = a_i$$
$$f_i = \text{RespCycle}(s_i, \text{instr}_i)$$
$$c_i = \max(f_i,\ c_{i-1})$$

Note however that unlike load instructions, store instructions are assigned constant execution
latencies in our trace analysis (we assume the architecture uses write-back with store forwarding).
So, despite using the same equations to define each instruction's timestamps, the throughput
distributions produced by the models could (and probably should) look quite different.

And the throughput for the $j$-th window of $k$ instructions is computed the same way:

$$\text{thr}_j^{\text{SQ}} = \frac{k}{c_{kj} - c_{k(j-1)}}$$

## gem5 Representation

We configure the store queue size of our core in gem5 by setting the already exposed `SQEntries` attribute of the
`X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Commit Width
## Description
A CPU's commit width is the maximum number of instructions that can be committed (i.e., have their results
be written to architectural state) in a single clock cycle. 

## Model Overview

The throughput bound imposed by the commit width is trivially the commit width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the commit width of our core in gem5 by setting the already exposed `commitWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Writeback Width
## Description
A CPU's writeback width is the maximum number of instructions that can update temporary storage (like physical 
registers) in a single clock cycle.

## Model Overview

The throughput bound imposed by the writeback width is trivially the writeback width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the writeback width of our core in gem5 by setting the already exposed `wbWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`. Originally, this was independently parameterized, however
initial simulations showed that simulations failed when the wbWidth was too small relative to the total issue width 
(sum of the issue widths of each functional unit type). Therefore, we currently set wbWidth = total issue width, 
so it can not be a bottleneck even when the CPU is hitting maximum issue throughput.

# Fetch Width
## Description
A CPU's fetch width is the maximum number of instructions that can be fetched from the I-cache in a
single clock cycle.

## Model Overview

The throughput bound imposed by the fetch width is trivially the fetch width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the fetch width of our core in gem5 by setting the already exposed `fetchWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Decode Width
## Description
A CPU's decode width is the maximum number of instructions that can be decoded into their corresponding
micro-instructions in a single clock cycle. 

## Model Overview

The throughput bound imposed by the decode width is trivially the decode width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the decode width of our core in gem5 by setting the already exposed `decodeWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Rename Width
## Description
A CPU's rename width is the maximum number of instructions that can be renamed (have their architectural
registers be mapped to physical registers to eliminate false dependencies) in a single clock cycle.  

## Model Overview

The throughput bound imposed by the rename width is trivially the rename width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the rename width of our core in gem5 by setting the already exposed `renameWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

# ALU Issue Width
## Description
A CPU's ALU issue width is the maximum number of ALU instructions that can be issued in a single clock
cycle.

## Model Overview

Although the ALU issue width is also a static bandwidth resource, it affects only the subset of
instructions that use the ALU. To find the throughput bound imposed by this parameter in isolation, we
treat all other types of instructions as completing instantaneously. Therefore, we only consider the
minimum time to process all ALU instructions in the given window.

For a window $j$ of $k$ consecutive instructions, the throughput bound is:

$$\text{thr}_j^{\text{ALU}} = \frac{k}{n_j^{\text{ALU}}} \times W_\text{ALU}$$

where $n_j^{\text{ALU}}$ is the number of ALU instructions in window $j$.

TODO: modify to separate Mult/Div instructions from normal ALU instructions.

## gem5 Representation

We configure the ALU issue width of our core in gem5 by setting the already exposed `instQueues` attribute of the `X86O3CPU` class in `configs/peregrine/peregrine.py`. This consists of an `IQUnit` with its `fuPool` set with a list of `FUDesc` objects. We configured our core to have separate issue widths for normal INT ALU commands and INT MULT/DIV ALU commands. The issue widths are configured with the `count` attribute of the `IntALU` and `IntMultDiv` functional units descriptions in the `fuPool` list.

# Floating-Point Issue Width
## Description
A CPU's floating-point issue width is the maximum number of floating point instructions that can be
issued in a single clock cycle.

## Model Overview

Although the floating-point issue width is also a static bandwidth resource, it affects only the subset of
instructions that use floating-point functional units. To find the throughput bound imposed by this
parameter in isolation, we treat all other types of instructions as completing instantaneously. Therefore,
we only consider the minimum time to process all floating-point instructions in the given window.

For a window $j$ of $k$ consecutive instructions:

$$\text{thr}_j^{\text{FP}} = \frac{k}{n_j^{\text{FP}}} \times W_\text{FP}$$

where $n_j^{\text{FP}}$ is the count of floating-point instructions in window $j$.

TODO: modify to separate FP Mult/Div instructions from normal FP ALU instructions.

## gem5 Representation

Similarly to normal ALU issue width, we configure the `fuPool` with `FUDesc` objects called `FP_ALU` and `FP_MultDiv`, with their `count` attributes set to the intended issue width.

# Load-Store Issue Width
## Description
A CPU's load-store issue width is the maximum number of memory instructions that can be issued in a single
clock cycle.

## Model Overview

Although the load-store issue width is also a static bandwidth resource, it affects only the subset of
instructions that interact with memory. To find the throughput bound imposed by this parameter in isolation,
we treat all other types of instructions as completing instantaneously. Therefore, we only consider the
minimum time to process all memory instructions in the given window.

For a window $j$ of $k$ consecutive instructions:

$$\text{thr}_j^{\text{LS}} = \frac{k}{n_j^{\text{LS}}} \times W_\text{LS}$$

where $n_j^{\text{LS}}$ is the total count of load and store instructions in window $j$.

TODO: account for load issue width and load-store issue width.

## gem5 Representation

Similarly to ALU and FP issue widths, for load-store issue width, we configure the `fuPool` with `FUDesc` objects called `ReadPort` (load issue width) and `RdWrPort` (load-store issue width), with their `count` attributes set to the intended issue width.

# Load and Load-Store Pipes
## Description
Load and Load-Store pipes are execution pipes that service memory instructions and crucially ensure that
memory dependencies are honored. Load pipes only service load instructions, while load-store pipes can service
both load and store instructions.

## Model Overview

The number of load pipes and number of load-store pipes are modelled **jointly** as a single dynamic constraint since they closely interact with each other (store instructions compete with load instructions for load-store
pipes). Rather than computing exact throughputs, this model computes upper and lower bounds on throughput corresponding to best-case and worst-case allocation, respectively.

For a window of $k$ consecutive instructions, let:

- $n_\text{Load}$ = number of load instructions
- $n_\text{Store}$ = number of store instructions
- $\text{LSP}$ = number of load-store pipes
- $\text{LP}$ = number of load pipes

### The Lower Bound: Worst-Case Allocation

The worst-case pipe allocation is to issue all loads first, using every available pipe (both load-store and load pipes), and only then begin issuing stores using the load-store pipes. This leaves the load pipes sitting idle during the store-issuing phase — a wasteful allocation that maximises total processing time.

The maximum total processing time under this allocation is:

$$T_\text{max} = \frac{n_\text{Load}}{\text{LSP} + \text{LP}} + \frac{n_\text{Store}}{\text{LSP}}$$

This gives a **lower bound on throughput**:

$$\text{thr}^\text{lower} = \frac{k}{T_\text{max}}$$

### The Upper Bound: Best-Case Allocation

The best-case allocation grants stores exclusive access to the load-store pipes while simultaneously using load pipes to issue loads in parallel. Once all stores have been issued, the load-store pipes become available to service any remaining loads alongside the load pipes.

This minimises idle pipe time and gives a **upper bound on throughput** $\text{thr}^\text{upper}$, derived analogously (the paper omits the explicit formula, noting it follows the same structure as the lower bound).

## gem5 Representation

We opted to only include load/store queues and load/store issue width.

# Fetch Buffers
## Description
The fetch buffer (sometimes called the fetch queue or instruction buffer) sits between the I-cache and the decode stage. It holds instructions that have been fetched from the I-cache and are waiting to be decoded. When the fetch buffer is full, the frontend stalls even if the I-cache is able to supply more instructions. Conversely, when the fetch buffer drains faster than the I-cache can refill it — for instance due to I-cache misses — the decode stage can be starved of instructions.

## Model Overview

TODO: still not really clear

## gem5 Representation

We opted to ignore this parameterization.

# Maximum I-Cache Fills
## Description
When the CPU fetches instructions, it may need to request cache lines from the I-cache. The maximum I-cache fills parameter caps how many such requests can be outstanding simultaneously. If this limit is reached, fetch stalls until an in-flight request completes and frees a slot.

## Model Overview

Maximum I-cache fills is a dynamic constraint because it depends on whether a given instruction generates a new
I-cache request, which depends on which requests are already in-flight (could be the same cache line) when the
instruction reaches the fetch target queue. To estimate the throughput the model performs a simulation as
follows:

1. Instructions are considered in program order, with a backlog assumed to always be waiting to be fetched — i.e. the simulation is never starved of instructions to process, so the only bottleneck it models is the availability of I-cache fill slots.
2. For each instruction, if its cache line is not already covered by an in-flight request, a new I-cache request is issued — but only once a fill slot is available. If all slots are occupied, the simulation waits until the earliest outstanding request completes.
3. The I-cache response cycle for each instruction is recorded based on when its cache line's request completes.
4. Using these response cycles, the throughput for each window of $k$ consecutive instructions is calculated the same way as in the ROB model — as $k$ divided by the difference in commit cycles between the start and end of the window.

The I-cache latency estimates used in the simulation come from the in-order I-cache simulation performed during trace analysis (§3.1), which is run per L1i/L2 cache size configuration.

## gem5 Representation

We control the number of in-flight requests the I-Cache can have by configuring the number of MSHRs (Miss-Status Hit Registers) in the i-cache system. The core's cache hierarchy is defined when the core processor and the cache hierarchy are connected together by the system board. The `L1ICache` object has a `mshrs` parameter that is configured to match the maximum i-cache fills constraint. 

# Cache Configuration
A CPU's cache configuration or cache hierarchy is how the processor interacts with the memory system. Our system consisted of a 2 level cache hierarchy, with a split L1 I-cache and D-cache, and a shared L2 cache. The L1 D-cache has a stride prefetcher. We parameterize the L1d cache size, L1i cache size, L2 cache size, and L1d stride prefetcher degree.

## Model Overview

The analytical cache model is implemented in `evantrace/caches.py`. Each cache is parameterized by line
size, total size, associativity, replacement policy, and read/write latency. A cache can have a parent
cache; misses are forwarded to the parent (or to main memory if there is no parent), and the returned
latency is the cache's own latency plus the parent's latency for misses and, when applicable, writeback
of a dirty victim. The trace is run through this hierarchy (I-cache for instruction addresses, D-cache
for load/store addresses) to compute per-instruction **fetch_latency** (I-cache read on the instruction
pointer) and **exec_latency** (opcode latency plus D-cache read or write latency for loads and stores).
These values are written into the instruction trace.

The anamol throughput models consume these trace fields. **fetch_latency** is used in the maximum
I-cache fills model and the fetch-buffers model. **exec_latency** (stored as `exe_latency` in the
trace) is used in `resp_cycle` in `models.cpp`: it determines the finish cycle of each instruction
and thus feeds the ROB, load-queue, and store-queue throughput models.

## gem5 Representation

The core's cache hierarchy can be parameterized directly when instantiated and connected to the core through the system board. We control the `l1d_size`, `l1i_size`, and `l2_size` arguments to the `PrivateL1SharedL2CacheHierarchy` object in `configs/peregrine/peregrine.py`. The `PrefetcherCls` argument to the `PrivateL1SharedL2CacheHierarchy` object is also set to the `StridePrefetcher` class in `configs/peregrine/peregrine.py`, where the `degree` argument is set to control the stride prefetching degree.

# Branch Predictor
The CPU's branch predictor controls the instruction fetching frontend of the out-of-order pipeline. There are various types of predictors that take inputs like the current PC, branch history, etc., to predict the next PC. This allows the front-end to fetch instructions after a branch instruction before it can even be evaluated later in the pipeline.

## Model Overview

The analytical branch predictor model is implemented in `evantrace/branch_predictor.py`. Implementations
include simple predictor configurations and a TAGE config. Each predictor exposes `predict(inst_ptr, branch_type)` → taken/not taken and `update(inst_ptr, branch_type, predicted_taken, actual_taken)`.

In `evantrace/sim.py`, the trace is run in order: for each instruction, `predicted_taken` is obtained
from the predictor's `predict`, then the predictor is updated with `predicted_taken` and the actual
outcome `instruction.branch_taken`. 

TODO: Use the branch misprediction rate and distribution as output features for the ML model training.

## gem5 Representation

We configure the branch predictor of our core in gem5 by setting the already exposed `branchPred` attribute of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

# How to export this file to a PDF

```bash
brew install pandoc # use pandoc to convert markdown to pdf
brew install mactex # use mactex to render the embedded latex

pandoc MODELS.md -o MODELS.pdf --pdf-engine=pdflatex
```
