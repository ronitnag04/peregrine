/*
 * adversarial_branches.c
 *
 * Blind spot: simple_model.py bins branches by TYPE (indirect,
 * direct_conditional, direct_unconditional) but not by PATTERN.
 * A workload where 99% of branches are direct_conditional with a
 * trivially predictable period-2 alternation looks identical, from
 * ronamol's perspective, to a workload where the same fraction of
 * direct_conditional branches follows a period that defeats
 * conventional predictors (e.g., TAGE's history length).
 *
 * This benchmark crafts a long alternation of three interleaved
 * branch streams with co-prime periods (5, 7, 11) so no reasonable
 * local history can align. The BP-rate sidecar sees it; features
 * alone do not.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef ITERS
/* Measured: ~47 dyn insts per iter. 200M iters -> ~9.4B dyn insts. */
#define ITERS (200000000ULL)
#endif

int main(void) {
    uint64_t acc = 0;

    /* Keep loop body tiny so branch density dominates the dynamic
     * instruction mix. Each iteration executes several
     * direct_conditional branches chosen to look innocuous.
     */
    for (uint64_t i = 0; i < ITERS; ++i) {
        /* Three coprime periodic sub-branches */
        if ((i % 5) < 3) acc += i;
        if ((i % 7) < 4) acc ^= i;
        if ((i % 11) < 6) acc -= (i >> 1);

        /* A data-derived branch whose condition rotates with a
         * period designed to be longer than any reasonable local
         * history table. 4096 is >> 64-entry local history.
         */
        if (((acc >> 17) & 0xFFF) > 2048) acc += 0xA5;
    }

    printf("adversarial_branches: iters=%lu periods=[5,7,11,4096] checksum=%lu\n",
           (unsigned long)ITERS, (unsigned long)acc);
    return 0;
}
