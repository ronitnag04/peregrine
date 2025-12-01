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
11. ??? TLB

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