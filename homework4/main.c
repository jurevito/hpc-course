#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// Total primes less or equal to 1000: 168
// Total primes less or equal to 10000: 1229
// Total primes less or equal to 100.000.000: result: 5761455 time: 0.762

void simple_sieve(int n) {
    //Initialize the array of flags for the primes
    unsigned char * primes = (unsigned char *)malloc((n+1) * sizeof(unsigned char));
    if (!primes) return;
    memset(primes, 1, (n+1));

    //Intialize array for the prime counts
    unsigned int * prime_cnts = (unsigned int *)calloc(omp_get_max_threads(), sizeof(unsigned int));
    double start = omp_get_wtime();
    //check for primes until sqrt(n)
    for (int p = 2; p * p <= n; p++)
    {

        //if flag is set then we encountered a prime number
        if (primes[p])
        {
            //cross out multiples of p grater than the square of p,
            //smaller have already been marked
            #pragma omp parallel for
            for (int i = p * p; i <= n; i += p)
                primes[i] = 0;
        }
    }
 
    //find and sum up all primes
    #pragma omp parallel for
    for (int p = 2; p <= n; p++)
        if (primes[p])
            prime_cnts[omp_get_thread_num()]++;
    
    unsigned int totalPrimes=0;
    for(int i=0; i<omp_get_max_threads(); i++)
    {
        totalPrimes+=prime_cnts[i];
    }
    double stop=omp_get_wtime();
    printf("Total primes less or equal to %d: %d\n",n,totalPrimes);
    printf("Elapsed time: %.3f\n",stop-start);
    free(primes);
    free(prime_cnts);
}

int sieve_of_eratosthenes(int n) {
    unsigned int total_primes = 0;
    int seg_size = (int)sqrt(n) + 1;

    unsigned char* primes = (unsigned char*)malloc(seg_size * sizeof(unsigned char));
    if (!primes) return 0;
    memset(primes, 1, seg_size);

    for (int p = 2; p * p < seg_size; p++) {
        if (primes[p]) {
            #pragma omp parallel for
            for (int i = p * p; i < seg_size; i += p) {
                primes[i] = 0;
            }
        }
    }

    #pragma omp parallel for reduction(+:total_primes)
    for (int p = 2; p <= seg_size; p++) {
        total_primes += primes[p];
    }

    printf("1.\n");
    
    //for (int i = 0; i < seg_size; i++)
    //    printf(", %d (%d)", primes[i], i);
    //printf("\n");
    #pragma omp parallel for reduction(+:total_primes)
    for(int low = seg_size ; low < n ; low+=seg_size) {
        //printf("n. primes: %d\n", total_primes);
        int high = (low + seg_size) > n ? n : (low + seg_size);
        // printf("%d -> %d\n", low, high);

        unsigned char* seg_primes = (unsigned char*)malloc((high - low) * sizeof(unsigned char));
        memset(seg_primes, 1, (high - low));

        for(int p = 2 ; p<seg_size ; p++) {
            if(primes[p]) {
                for(int j = 0 ; j<(high - low) ; j++) {
                    //printf("%d divisible by %d = %d\n", (low + j), p, (low + j) % p);
                    if((low + j) % p == 0) {
                        seg_primes[j] = 0;
                    }
                }
            }
        }

        //for (int i = 0; i < (high - low); i++)
        //    printf(", %d (%d)", seg_primes[i], low + i);
        //printf("\n");
        //printf("3.\n");
        for (int p = 0; p < (high - low); p++) {
            total_primes += seg_primes[p];
        }
        //printf("4. %d %p\n", (high - low), seg_primes);
        //printf("(%d -> %d) n. primes: %d\n", low, high, total_primes);
        //printf("4.\n");
        free(seg_primes);
        //printf("5.\n");
    }

    /*
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
    */
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

    simple_sieve(N);
    return 0;
}
