/*
 * ptrchase_rand.c
 *
 * Blind spot: simple_model.py cannot measure memory-level parallelism.
 * frac_mem_dependent / frac_memory_ordering_hazards count only
 * whether a memory op has a predecessor -- they cannot distinguish:
 *   (a) a tight serial chain where every load waits for the prior
 *       load's result (MLP = 1), and
 *   (b) a random-gather kernel where many loads are in flight.
 *
 * This benchmark is (a). Paired with `many_pages_streaming.c` tuned
 * to the same unique_load_pages, the two look near-identical in
 * feature space but have CPI differing by >10x.
 *
 * Scaling: default HOPS = 10B. Each hop is ~3 dyn insts (load, add,
 * branch), for ~30B dyn insts. Override via -DHOPS=... or env.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef NODES
#define NODES (1u << 22)       /* 4M nodes at 64B/line = 256 MB */
#endif

#ifndef HOPS
/* Measured: ~5.7 dyn insts per hop. User wants ~2B dyn insts for
 * this benchmark (it's latency-bound and dominates wall time).
 * 350M hops -> ~2B dyn insts.
 */
#define HOPS (350000000ULL)
#endif

#define CACHELINE 64

typedef struct node {
    struct node *next;
    uint64_t pad[CACHELINE/sizeof(uint64_t) - 1];
} node_t;

static void build_cycle(node_t *arr, uint32_t n) {
    uint32_t *perm = malloc((size_t)n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; ++i) perm[i] = i;

    uint64_t s = 0xC0FFEE1234ULL;
    for (uint32_t i = n - 1; i > 0; --i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint32_t j = (uint32_t)(s % (uint64_t)(i + 1));
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (uint32_t i = 0; i < n; ++i) {
        arr[perm[i]].next = &arr[perm[(i + 1) % n]];
    }
    free(perm);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    uint64_t hops = HOPS;
    const char *env = getenv("PTRCHASE_HOPS");
    if (env) hops = strtoull(env, NULL, 0);

    node_t *arr = aligned_alloc(4096, (size_t)NODES * sizeof(node_t));
    if (!arr) return 1;
    memset(arr, 0, (size_t)NODES * sizeof(node_t));

    build_cycle(arr, NODES);

    node_t *p = &arr[0];
    uint64_t acc = 0;
    /* Serial chase: every load depends on the previous load's result.
     * MLP = 1, but ronamol reports only frac_mem_dependent ~= 1.0,
     * identical to what a parallel gather kernel would produce.
     */
    for (uint64_t i = 0; i < hops; ++i) {
        acc += (uintptr_t)p;
        p = p->next;
    }

    printf("ptrchase_rand: nodes=%u node_bytes=%zu working_set_bytes=%zu hops=%lu final_ptr=%p addr_sum=%lu\n",
           (unsigned)NODES, sizeof(node_t),
           (size_t)NODES * sizeof(node_t),
           (unsigned long)hops, (void *)p, (unsigned long)acc);
    free(arr);
    return 0;
}
