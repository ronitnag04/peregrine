### Ronamol program features (alphabetical)

This document explains each field in `program_features.json` as computed by `compute_program_features` in `simple_model.py`. The explanations are ordered alphabetically by feature name to match the JSON output.

- **branch_rate**: Fraction of dynamic instructions that are branches (`n_branch / total`), where a branch is any instruction with `inst.branch_type` not `None`. This is numerically the same as `frac_branch` but grouped as a frontend/control‑flow feature.

- **crit_path_density_10**: Fraction of dynamic instructions whose maximum register dependence distance (the farthest producer in `reg_dependent_ips` measured in dynamic instruction index) exceeds the critical threshold of 10. This serves as a proxy for the density of long‑latency dependences and thus critical‑path intensity.

- **frac_branch**: Fraction of dynamic instructions that are branches (`n_branch / total`). This measures how branch‑heavy the instruction stream is and directly affects frontend and branch‑prediction behavior.

- **frac_direct_cond_branches**: Among all branches, the fraction that are direct conditional branches (`branch_type == "direct_conditional"`). These are the typical data‑dependent control‑flow branches (e.g., loop conditions, if‑statements).

- **frac_direct_uncond_branches**: Among all branches, the fraction that are direct unconditional branches (`branch_type == "direct_unconditional"`). These are jumps/calls/returns that do not depend on a runtime condition.

- **frac_fp_alu**: Fraction of dynamic instructions that are floating‑point ALU operations (`fu_group == "fp_alu"`). Indicates the intensity of scalar floating‑point arithmetic and logical operations.

- **frac_fp_mult_div**: Fraction of dynamic instructions that are floating‑point multiply/divide operations (`fu_group == "fp_mult_div"`). These are typically higher‑latency FP operations (e.g., multiplication, division, square root).

- **frac_independent_last_16**: Fraction of dynamic instructions whose maximum register dependence distance is either nonexistent (no dependencies found) or greater than the dependency window of 16 instructions. Implemented by incrementing `indep_count` when `max_dist is None` or `max_dist > dep_window`, then dividing by total instructions. This approximates the fraction of instructions that are “independent of the last 16” and thus available for ILP within a small scheduling window.

- **frac_indirect_branches**: Among all branches, the fraction that are indirect branches (`branch_type == "indirect"`). These represent harder‑to‑predict control flow, such as function pointers, virtual calls, or computed jumps.

- **frac_int_alu**: Fraction of dynamic instructions that are integer ALU operations (`fu_group == "int_alu"`). Captures how much work is simple integer arithmetic and logic.

- **frac_int_mult_div**: Fraction of dynamic instructions that are integer multiply/divide operations (`fu_group == "int_mult_div"`). These tend to be more expensive integer instructions and can affect ALU latency/throughput.

- **frac_load**: Fraction of dynamic instructions that are loads (`fu_group == "read_port"`). This reflects demand on the load side of the memory subsystem and influences cache/memory pressure.

- **frac_mem_dependent**: Among all memory operations (any instruction with at least one load or store address), the fraction that have at least one memory dependence (`inst.mem_dependent_ips` non‑empty). This measures how often memory operations are constrained by prior memory accesses (e.g., true/anti/output dependencies).

- **frac_memory_ordering_hazards**: Among all memory operations, the fraction that have more than one memory dependence (`len(inst.mem_dependent_ips) > 1`). Used as a proxy for complex memory ordering hazards that can stress reordering and consistency mechanisms.

- **frac_other**: Fraction of dynamic instructions whose functional unit group is labeled `"other"`. This is a catch‑all bucket for instructions that do not fall into the main ALU, FP, SIMD, or explicit memory categories.

- **frac_simd**: Fraction of dynamic instructions that are SIMD/vector operations (`fu_group == "simd_unit"`). Indicates how vectorized the workload is, which affects SIMD unit utilization and potential throughput.

- **frac_store**: Fraction of dynamic instructions that are stores (`fu_group == "rdwr_port"`). Reflects demand on the store/write side of the memory hierarchy and write bandwidth.

- **frac_sync_instructions**: Fraction of dynamic instructions marked as synchronization operations (`inst.inst_sync` true). These include fences/barriers and other instructions that serialize or coordinate memory/threads, often limiting reordering.

- **load_stride_regularity**: For each static load instruction (identified by instruction pointer) with at least `stride_min_samples` dynamic instances (default 16), the model tracks the distribution of address deltas between consecutive loads. For that load, it computes `best / total_deltas`, where `best` is the most frequent delta count and `total_deltas` is the sum over all observed deltas. `load_stride_regularity` is the mean of these per‑IP regularity scores. Higher values indicate many loads follow a dominant, regular stride (e.g., streaming through arrays), which benefits prefetching and cache locality.

- **mean_basic_block_size**: Average number of instructions per basic block, where a basic block is terminated by a taken branch. Computed as the mean of `bb_sizes`. This characterizes straight‑line code length and effective branch density.

- **mean_reg_dep_distance**: Mean dynamic register dependence distance. For each instruction, the model looks at `reg_dependent_ips`, maps each producer IP to the last dynamic index where it appeared, and computes the distance `idx - prev`. The maximum distance per consumer is tracked, and all such distances form `reg_dep_dists`. `mean_reg_dep_distance` is the average of this array and reflects typical producer‑to‑consumer spacing in the dynamic stream.

- **mean_reg_fan_out**: Average fan‑out of producer instructions. For each producer instruction pointer, the model counts how many dependent uses reference it (`producer_fanout`), then divides the sum of those counts by the number of producers. This indicates how widely results are reused across consumers.

- **memory_instruction_fraction**: Fraction of dynamic instructions that are memory operations (any instruction with at least one read or write address), computed as `(n_load + n_store) / total`. This is separate from the functional‑unit buckets and expresses how memory‑centric the instruction stream is.

- **p50_basic_block_size**: Median basic block size, i.e., the 50th percentile of `bb_sizes`. Represents a “typical” number of instructions between taken branches.

- **p50_reg_dep_distance**: 50th percentile (median) of the register dependence distances stored in `reg_dep_dists`. Indicates a typical producer‑to‑consumer gap in dynamic instructions.

- **p95_basic_block_size**: 95th percentile of basic block sizes. Captures the tail of unusually long straight‑line regions, which can impact frontend utilization and I‑cache behavior.

- **p95_reg_dep_distance**: 95th percentile of register dependence distances. Highlights the presence and severity of long‑range dependences that may dominate the critical path.

- **store_to_load_ratio**: Ratio of store to load counts, computed as `n_store / n_load` with safe division. Indicates how write‑heavy versus read‑heavy the memory traffic is.

- **unique_load_pages**: Number of distinct virtual memory pages (4 KiB, via `addr >> 12`) touched by load instructions. This is a coarse measure of the spatial footprint and working‑set spread of read traffic.

- **unique_store_pages**: Number of distinct virtual memory pages touched by store instructions. Similar to `unique_load_pages` but for writes, it helps characterize the write working set.
