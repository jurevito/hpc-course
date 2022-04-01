#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Total primes less or equal to 1000: 168
// Total primes less or equal to 10000: 1229
// Total primes less or equal to 100000000: result: 5761455 time: 0.762
int sieve_of_eratosthenes(int n) {

    // Initialize the array of flags for the primes.
    unsigned int* primes = (unsigned int*)malloc((n + 1) * sizeof(unsigned int));
    for (int i = 0; i < n + 1; i++)
        primes[i] = 1;

    // check for primes until sqrt(n)
    for (int p = 2; p * p <= n; p++) {

        // if flag is set then we encountered a prime number
        if (primes[p]) {
            // cross out multiples of p grater than the square of p,
            // smaller have already been marked
            #pragma omp parallel for
            for (int i = p * p; i <= n; i += p) {
                primes[i] = 0;
            }
        }
    }

    // Count number of primes.
    unsigned int total_primes = 0;
    #pragma omp parallel for reduction(+:total_primes)
    for (int p = 2; p <= n; p++) {
        total_primes += primes[p];
    }

    free(primes);
    return total_primes;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Not enough arguments!\n");
        printf("Usage: sieve <N>!\n");
        return 1;
    }
    unsigned int N = atoi(argv[1]);

    // Measure function performance.
    double start_time = omp_get_wtime();
    int total_primes = sieve_of_eratosthenes(N);
    double elapsed_time = omp_get_wtime() - start_time;
    printf("result: %d time: %.3lf\n", total_primes, elapsed_time);

    return 0;
}
