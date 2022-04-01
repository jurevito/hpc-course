#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Total primes less or equal to 1000: 168
// Total primes less or equal to 10000: 1229

int sieve_of_eratosthenes(int n) {

    // Initialize the array of flags for the primes.
    unsigned char* primes = (unsigned char*)malloc((n + 1) * sizeof(unsigned char));
    if (!primes) {
        return 0;
    }
    memset(primes, 1, (n + 1));
    /*
    for(int i = 0 ; i<(n + 1) ; i++) {
        printf(", %d", primes[i]);
    }
    printf("\n");
    */
    // Intialize array for the prime counts
    unsigned int* prime_cnts = (unsigned int*)calloc(omp_get_max_threads(), sizeof(unsigned int));
    
    // check for primes until sqrt(n)
    for (int p = 2; p * p <= n; p++) {

        // if flag is set then we encountered a prime number
        if (primes[p]) {
            // cross out multiples of p grater than the square of p,
            // smaller have already been marked
            #pragma omp parallel for
            for (int i = p * p; i <= n; i += p)
                primes[i] = 0;
        }
    }

    // find and sum up all primes
    #pragma omp parallel for
    for (int p = 2; p <= n; p++)
        if (primes[p])
            prime_cnts[omp_get_thread_num()]++;

    unsigned int totalPrimes = 0;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        totalPrimes += prime_cnts[i];
    }

    free(primes);
    free(prime_cnts);

    return totalPrimes;
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
