#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "/usr/include/openmpi-x86_64/mpi.h"

/*
Number of Samples: 50.000.000
Serial: 4.621
┌─────────────┬───────────┬────────────┬─────────┐
│ n. threads  │ n. nodes  │ send/recv  │ reduce  │
├─────────────┼───────────┼────────────┼─────────┤
│ 1           │ 1         │ 4.620      │ 4.620   │
│ 2           │ 1         │ 2.319      │ 2.318   │
│ 2           │ 2         │ 2.310      │ 2.297   │
│ 4           │ 1         │ 1.180      │ 1.161   │
│ 4           │ 2         │ 1.154      │ 1.151   │
│ 8           │ 1         │ 0.602      │ 0.605   │
│ 8           │ 2         │ 0.607      │ 0.658   │
│ 16          │ 1         │ 0.301      │ 0.301   │
│ 16          │ 2         │ 0.386      │ 0.335   │
│ 32          │ 1         │ 0.862      │ 0.147   │
│ 32          │ 2         │ 0.269      │ 0.148   │
└─────────────┴───────────┴────────────┴─────────┘
*/

#define SEED 42
#define SAMPLES 50000000
#define PI 3.14159265359

double monte_carlo_pi_serial(int n_samples, int seed) {

    int count = 0;
    double x, y, z;

    srand(seed + time(0));
    for (int i = 0; i < n_samples; i++) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        if (z <= 1.0) {
            count++;
        }
    }

    // Calculate π.
    return ((double)count / n_samples) * 4.0;
}

double monte_carlo_pi_parallel(int n_samples, int seed) {

    int id, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    int count = 0;
    double x, y, z;

    srand(id + time(0));
    for (int i = 0; i < (n_samples / n_processes); i++) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        if (z <= 1.0) {
            count++;
        }
    }

    int count_buffer = count;
    double pi;

    if (id == 0) {
        MPI_Status status;
        int total_count = count;

        for (int i = 1; i < n_processes; i++) {
            MPI_Recv(&count_buffer, sizeof(int), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            total_count += count_buffer;
        }

        pi = ((double)total_count / n_samples) * 4.0;
    } else {
        MPI_Send(&count_buffer, sizeof(int), MPI_INT, 0, id, MPI_COMM_WORLD);
    }

    return pi;
}

double monte_carlo_pi_reduce(int n_samples, int seed) {

    int id, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    int count = 0;
    double x, y, z;

    srand(id + time(0));
    for (int i = 0; i < (n_samples / n_processes); i++) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        if (z <= 1.0) {
            count++;
        }
    }

    int input = count;
    int output = 0;

    MPI_Reduce(&input, &output, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi = 0;

    if (id == 0) {
        pi = ((double)output / n_samples) * 4.0;
    }

    return pi;
}

int main(int argc, char* argv[]) {

    int provided_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_level);

    double start_time, elapsed_time, pi;
    int is_correct;

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_serial(SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Serial:   pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_parallel(SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Parallel: pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_reduce(SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Reduce:   pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    MPI_Finalize();
    return 0;
}