/*
 * stlf_misalign.c
 *
 * Blind spot: simple_model.py flags a "memory ordering hazard" only
 * when len(mem_dependent_ips) > 1. A canonical STLF-busting pattern
 * has exactly one prior store feeding one load, so
 * frac_memory_ordering_hazards = 0 even though every load stalls
 * for the store buffer to drain.
 *
 * The store is a 1-byte write into offset 0 of a qword that is
 * subsequently loaded as an 8-byte load. On all mainstream cores
 * (x86 and AArch64) the size/alignment mismatch prevents
 * store-to-load forwarding, forcing a store-queue flush or a
 * replay. The analytical model sees "a single benign producer."
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef ITERS
/* Measured: ~6 dyn insts per iter. 1.66B iters -> ~10B dyn insts. */
#define ITERS (1660000000ULL)
#endif

#define BUFSZ 1024

int main(void) {
    uint8_t *buf = aligned_alloc(64, BUFSZ);
    if (!buf) return 1;
    memset(buf, 0, BUFSZ);

    uint64_t acc = 0;

    /* volatile pointers so the compiler keeps the store->load
     * round-trip and doesn't substitute a register forward.
     */
    volatile uint8_t  *pb = buf;
    volatile uint64_t *pq = (volatile uint64_t *)buf;

    for (uint64_t i = 0; i < ITERS; ++i) {
        /* Narrow store to offset 0 */
        pb[0] = (uint8_t)(i & 0xFF);
        /* Wider load overlapping the narrow store -> STLF-stall */
        uint64_t v = pq[0];
        acc += v;
    }

    printf("stlf_misalign: iters=%lu store_bytes=1 load_bytes=8 overlap=1 checksum=%lu\n",
           (unsigned long)ITERS, (unsigned long)acc);
    free(buf);
    return 0;
}
