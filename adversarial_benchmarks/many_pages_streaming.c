/*
 * many_pages_streaming.c
 *
 * Blind spot: simple_model.py reports `unique_load_pages` as a raw
 * count with no structural information about access order. A
 * pointer chase over P pages and a sequential scan over P pages
 * produce similar `unique_load_pages` values. But the streaming
 * version is prefetcher-friendly and lives near peak bandwidth,
 * while the pointer chase is latency-bound.
 *
 * Tune NUM_PAGES so `unique_load_pages` matches the
 * `ptrchase_rand.c` companion.
 *
 * Scaling: default PASSES = 40, giving each pass ~256M cache-line
 * reads; at ~3 insts per iter that's ~3.8B dyn insts per pass,
 * ~150B total across all passes. Use env MPS_PASSES to dial down.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef NUM_PAGES
#define NUM_PAGES (1u << 16)         /* 65536 pages -> 256 MiB */
#endif

#ifndef PASSES
/* Measured: ~1.20B dyn insts at PASSES=40 -> ~30M dyn insts per pass.
 * 333 passes -> ~10B dyn insts.
 */
#define PASSES 333u
#endif

#define PAGE      4096u
#define NUM_BYTES ((size_t)NUM_PAGES * (size_t)PAGE)

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    uint64_t passes = PASSES;
    const char *env = getenv("MPS_PASSES");
    if (env) passes = strtoull(env, NULL, 0);

    uint8_t *buf = aligned_alloc(PAGE, NUM_BYTES);
    if (!buf) return 1;

    /* Touch every page up-front so TLB/working set matches the
     * pointer-chase companion.
     */
    for (size_t i = 0; i < NUM_BYTES; i += PAGE) buf[i] = (uint8_t)i;

    uint64_t sum = 0;
    for (uint64_t p = 0; p < passes; ++p) {
        /* Sequential cache-line stride -- HW prefetcher hides latency. */
        for (size_t i = 0; i < NUM_BYTES; i += 64) {
            sum += buf[i];
        }
    }

    printf("many_pages_streaming: num_pages=%u page_bytes=%u working_set_bytes=%zu passes=%lu byte_sum=%lu\n",
           (unsigned)NUM_PAGES, (unsigned)PAGE, (size_t)NUM_BYTES,
           (unsigned long)passes, (unsigned long)sum);
    free(buf);
    return 0;
}
