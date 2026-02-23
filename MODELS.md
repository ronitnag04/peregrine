# Analytical Model Specifications

## How to export

```bash
brew install pandoc # use pandoc to convert markdown to pdf
brew install mactex # use mactex to render the embedded latex

pandoc MODELS.md -o MODELS.pdf --pdf-engine=pdflatex
```

## Table of Contents
1. [Re-order Buffer](#re-order-buffer)
2. [Load Queue](#load-queue)
3. [Store Queue](#store-queue)
4. [Commit Width](#commit-width)
5. [Fetch Width](#fetch-width)
6. [Rename Width](#rename-width)
7. [ALU Issue Width](#alu-issue-width)
8. [Floating-Point Issue Width](#floating-point-issue-width)
9. [Load-Store Issue Width](#load-store-issue-width)
10. [Load and Load-Store Pipes](#load-and-load-store-pipes)
11. [Maximum I-Cache Fills](#maximum-i-cache-fills)
12. [Fetch Buffers](#fetch-buffers)

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

**Equation 3 — Execution finish time.** The finish cycle is computed using a function called `RespCycle`, which is a small state machine that models memory behaviour (see appendix). For non-load instructions it simply adds a fixed latency; for load instructions it accounts for cache-line contention and issue ordering (see below).

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

We configure the ROB size of our core in gem5 by setting the already exposed `LQEntries` attribute of the
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

$$a_i = c_{i - \text{LQ}}$$
$$s_i = a_i$$
$$f_i = \text{RespCycle}(s_i, \text{instr}_i)$$
$$c_i = \max(f_i,\ c_{i-1})$$

Note however that unlike load instructions, store instructions are assigned constant execution
latencies in our trace analysis (we assume the architecture uses write-back with store forwarding).
So, despite using the same equations to define each instruction's timestamps, the throughput
distributions produced by the models could (and probably should) look quite different.

And the throughput for the $j$-th window of $k$ instructions is computed the same way:

$$\text{thr}_j^{\text{LQ}} = \frac{k}{c_{kj} - c_{k(j-1)}}$$

## gem5 Representation

We configure the ROB size of our core in gem5 by setting the already exposed `SQEntries` attribute of the
`X86O3CPU` class in `configs/peregrine/peregrine.py`.

# Commit Width
## Description
A CPU's commit width is the maximum number of instructions that can be committed (i.e. have their results
be written to registers/memory) in a single clock cycle. 

## Model Overview

The throughput bound imposed by the commit width is trivially the commit width itself, as it uniformly
impacts all instructions processed by the CPU.

## gem5 Representation

We configure the commit width of our core in gem5 by setting the already exposed `commitWidth` attribute
of the `X86O3CPU` class in `configs/peregrine/peregrine.py`.

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

## gem5 Representation

TODO

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

## gem5 Representation

TODO

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

## gem5 Representation

TODO

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

Opted to only include load/store queues and load/store issue width.

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

We can configure this parameter in our gem5 CPU by setting the MSHR count in the gem5 cache object.

# Fetch Buffers
## Description
The fetch buffer (sometimes called the fetch queue or instruction buffer) sits between the I-cache and the decode stage. It holds instructions that have been fetched from the I-cache and are waiting to be decoded. When the fetch buffer is full, the frontend stalls even if the I-cache is able to supply more instructions. Conversely, when the fetch buffer drains faster than the I-cache can refill it — for instance due to I-cache misses — the decode stage can be starved of instructions.

## Model Overview

TODO: still not really clear

## gem5 Representation

Ignore for now.


