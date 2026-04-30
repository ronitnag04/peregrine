/*
 * serial_mul_chain.c
 *
 * Blind spot: simple_model.py uses
 *   reg_dep_dists = [idx - last_seen_ip[producer_ip]]
 * For a tight inner loop whose body is one static multiply and one
 * static add, the producer is always the *immediately preceding*
 * dynamic instruction, so every dep distance is ~1:
 *   mean_reg_dep_distance  ~= 1
 *   p50_reg_dep_distance   == 1
 *   p95_reg_dep_distance   == 1
 *   crit_path_density_10   == 0       (threshold is > 10)
 *
 * The model therefore labels this as "short, harmless dependences."
 * In fact this is the worst ILP case: every op serializes on the
 * previous op. An OOO core with 4-wide dispatch runs at ~1 IPC
 * instead of the ~4 IPC the feature set implies.
 *
 * Scaling: default ITERS = 4B. Each iter is ~2 dyn insts (mul+add)
 * plus the loop branch, giving ~12B dyn insts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef ITERS
/* Measured: ~5 dyn insts per iter. 2B iters -> ~10B dyn insts. */
#define ITERS (2000000000ULL)
#endif

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    uint64_t iters = ITERS;
    const char *env = getenv("SERIAL_CHAIN_ITERS");
    if (env) iters = strtoull(env, NULL, 0);

    uint64_t x = 0x123456789abcdef0ULL;
    uint64_t k = 6364136223846793005ULL;
    uint64_t c = 1442695040888963407ULL;

    /* Single long dependence chain: each multiply depends on the
     * previous multiply's result.
     */
    for (uint64_t i = 0; i < iters; ++i) {
        x = x * k + c;
    }

    printf("serial_mul_chain: iters=%lu final_x=0x%016lx\n",
           (unsigned long)iters, (unsigned long)x);
    return 0;
}
