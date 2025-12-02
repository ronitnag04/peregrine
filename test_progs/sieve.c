#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int n = strtol(argv[1], NULL, 10);

    int* marked = calloc(n, sizeof(int));

    int smallest_non_marked = 2;
    int p, i;

    while (i != n) {
        p = 2 * smallest_non_marked;
        while (p < n) {
            marked[p] = p;
            p += smallest_non_marked;
        }

       for (i = smallest_non_marked + 1; i < n; i++) {
           if (marked[i] == 0) {
                smallest_non_marked = i;
                i = n + 1;
                break;
            }
        }
    }

    int prime_count = 0;
    for (i = 2; i < n; i++) {
        if (marked[i] == 0) {
            prime_count++;
        }        
    }
    
    free(marked);
    printf("Number of prime numbers below %d: %d\n", n, prime_count);

    return 0;
}
