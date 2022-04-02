#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
┌─────────────────┬────────┬────────┬────────┬────────┬────────┐
│ n. threads      │ 1      │ 4      │ 8      │ 16     │ 32     │
├─────────────────┼────────┼────────┼────────┼────────┼────────┤
│ execution time  │ 8.941  │ 2.270  │ 1.152  │ 0.578  │ 0.456  │
│ speed up        │ 1.00x  │ 3.94x  │ 7.76x  │ 15.47x │ 19.61x │
└─────────────────┴────────┴────────┴────────┴────────┴────────┘
*/

#define N_TIMES 20

int sieve_of_eratosthenes(int n) {

    unsigned int total_primes = 0;
    int seg_size = (int)sqrt(n) + 1;

    // Intialize array of prime flags for first segment.
    unsigned char* primes = (unsigned char*)malloc(seg_size * sizeof(unsigned char));
    if (!primes) return 0;
    memset(primes, 1, seg_size);

    // Find primes in first segment until sqrt(n).
    for (int p = 2; p * p < seg_size; p++) {
        if (primes[p]) {
            #pragma omp parallel for
            for (int i = p * p; i < seg_size; i += p) {
                primes[i] = 0;
            }
        }
    }

    // Count number of primes in first segment.
    #pragma omp parallel for reduction(+:total_primes)
    for (int p = 2; p < seg_size; p++) {
        total_primes += primes[p];
    }

    // Find primes for other segments.
    #pragma omp parallel for reduction(+:total_primes) schedule(guided)
    for (int low = seg_size; low < n; low += seg_size) {
        int high = (low + seg_size) > n ? n + 1 : (low + seg_size);

        unsigned char* seg_primes = (unsigned char*)malloc((high - low) * sizeof(unsigned char));
        memset(seg_primes, 1, (high - low));

        // Uncheck numbers that aren't primes.
        for (int p = 2; p < seg_size; p++) {
            if (primes[p]) {
                int start = !(low % p) ? 0 : p - low % p;

                for (int j = start; j < (high - low); j += p) {
                    seg_primes[j] = 0;
                }
            }
        }

        for (int p = 0; p < (high - low); p++) {
            total_primes += seg_primes[p];
        }

        free(seg_primes);
    }

    return total_primes;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Not enough arguments!\n");
        return 1;
    }
    unsigned int N = atoi(argv[1]);

    // Measure function performance.
    double average_time = 0;
    int total_primes = 0;
    for (int i = 0; i < N_TIMES; i++) {
        double start_time = omp_get_wtime();
        total_primes = sieve_of_eratosthenes(N);
        average_time += omp_get_wtime() - start_time;
    }

    printf("result: %d time: %.3lf\n", total_primes, average_time / N_TIMES);
    return 0;
}
