/*
 * icache_blast.c
 *
 * Blind spot: simple_model.py characterizes only the *dynamic*
 * instruction mix. There is no feature for the size of the static
 * code footprint. This benchmark dispatches pseudo-randomly through
 * hundreds of distinct tiny callees; the dynamic mix is boring
 * (mostly int ALU + one indirect branch + one return per iter),
 * but the footprint easily overflows the L1I.
 *
 * Scaling: default ITERS = 200M iterations, each iteration executes
 * ~60 dynamic instructions (callee body + dispatch), giving
 * ~12B dynamic instructions. Override with -DITERS=... or env
 * ICACHE_BLAST_ITERS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "icache_blast_gen.h"

#ifndef ITERS
/* Measured: ~29 dyn insts per iter at N_FUNCS=512, PAD_OPS=12.
 * 345M iters -> ~10B dyn insts.
 */
#define ITERS 345000000ULL
#endif

static uint32_t next_target(uint32_t state, uint32_t iter) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    state += iter * 2654435761u;
    return state;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    uint64_t iters = ITERS;
    const char *env = getenv("ICACHE_BLAST_ITERS");
    if (env) iters = strtoull(env, NULL, 0);

    uint32_t acc   = 0xDEADBEEFu;
    uint32_t state = 0x12345678u;

    for (uint64_t i = 0; i < iters; ++i) {
        state = next_target(state, (uint32_t)i);
        uint32_t idx = state % (uint32_t)ICACHE_BLAST_N_FUNCS;
        /* one indirect branch per iteration */
        acc ^= fn_table[idx](acc + (uint32_t)i);
    }

    printf("icache_blast: n_funcs=%u iters=%lu checksum=0x%08x\n",
           (unsigned)ICACHE_BLAST_N_FUNCS, (unsigned long)iters, acc);
    return 0;
}
