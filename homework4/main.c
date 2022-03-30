#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void sieve_of_eratosthenes(int n)
{
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

int main(int argc,char* argv[])
{
    if(argc < 2){
        printf("Not enough arguments!\n");
        printf("Usage: sieve <N>!\n");
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    sieve_of_eratosthenes(N);
    return 0;
}
