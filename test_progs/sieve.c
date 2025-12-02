<<<<<<< HEAD
#include <stdlib.h>
#include <stdio.h>




/* Implements the Sieve of Eratosthenes algorithm
 * for computing the number of primes below a
 * given number n. */
int sieve_of_eratosthenes(int n) {
  int* marked = calloc(n, sizeof(int)); // marked array
  
  int smallest_non_marked = 2; // smallest prime
  int p, i;

  while (i != n) {
    p = 2 * smallest_non_marked;
    while (p < n) {
      marked[p] = smallest_non_marked;
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
  return prime_count;
}

/* Main entry point */
int main(int argc, char* argv[]) {
  int n = strtol(argv[1], NULL, 10); // convert command line arg to number
  
  int prime_count = sieve_of_eratosthenes(n);

  printf("Number of primes below %d: %d\n", n, prime_count);
  return 0;
=======
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
>>>>>>> mother
}
