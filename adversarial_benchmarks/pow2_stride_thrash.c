/*
 * pow2_stride_thrash.c
 *
 * Blind spot: load_stride_regularity in simple_model.py averages, per
 * static load IP, (most_frequent_delta_count / total_deltas). A single
 * constant stride produces score 1.0 -- the max. The feature cannot
 * encode stride *magnitude* relative to cache geometry. This benchmark
 * picks a power-of-two stride equal to a typical (L1D_size / assoc)
 * so every access lands in the same L1D set.
 *
 * Compile two variants from this same file:
 *   -DSTRIDE_LOG2=15  -> 32 KiB stride -> L1D conflict thrash
 *   -DSTRIDE_LOG2=6   -> 64 B   stride -> sequential, cache-friendly
 *
 * Both variants produce essentially identical ronamol feature
 * vectors but wildly different CPI.
 *
 * Scaling: the outer pass count PASSES scales the dynamic instruction
 * count. At NUM_LINES=2048 and roughly 4 insts per iteration the
 * inner loop is ~8192 insts; PASSES=262144 => ~2.1B dynamic insts,
 * well within SPEC-region sampling territory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef STRIDE_LOG2
#define STRIDE_LOG2 15
#endif

#ifndef NUM_LINES
#define NUM_LINES 2048u
#endif

#ifndef PASSES
/* Measured: ~2.79B dyn insts at PASSES=262144 -> ~10650 insts per pass.
 * 940000 passes -> ~10B dyn insts. Both compile-variants share this
 * source, so the count applies to thrash and benign equally.
 */
#define PASSES 940000u
#endif

#define STRIDE      ((size_t)1u << STRIDE_LOG2)
#define BUF_BYTES   ((size_t)STRIDE * (size_t)NUM_LINES)

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    volatile uint8_t *buf = aligned_alloc(4096, BUF_BYTES);
    if (!buf) return 1;
    memset((void *)buf, 1, BUF_BYTES);

    uint64_t sum = 0;
    for (uint32_t p = 0; p < PASSES; ++p) {
        /* Single static load IP. Constant stride across dynamic iters.
         * stride_stats_by_ip[ip] collapses to one delta, so
         * load_stride_regularity for this IP is exactly 1.0.
         */
        for (size_t i = 0; i < NUM_LINES; ++i) {
            sum += buf[i * STRIDE];
        }
    }

    printf("pow2_stride_thrash: stride_log2=%u stride_bytes=%zu num_lines=%u passes=%u checksum=%lu\n",
           (unsigned)STRIDE_LOG2, (size_t)STRIDE, (unsigned)NUM_LINES,
           (unsigned)PASSES, (unsigned long)sum);
    free((void *)buf);
    return 0;
}
