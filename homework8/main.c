#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "/usr/include/openmpi-x86_64/mpi.h"

#define SEED 42
#define SAMPLES 50000000 // 50.000.000
#define PI 3.14159265359

double monte_carlo_pi_serial(int n_samples, int seed) {
    
    int count = 0;
    double x, y, z;

    srand(seed);
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

double monte_carlo_pi_parallel(int argc, char* argv[], int n_samples, int seed) {
    
    int provided_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_level);

    int id, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    char node_name[MPI_MAX_PROCESSOR_NAME];
    int node_name_len;
    MPI_Get_processor_name(node_name, &node_name_len);

    int count = 0;
    double x, y, z;

    srand(id);
    for (int i = 0; i < (n_samples / n_processes); i++) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        if (z <= 1.0) {
            count++;
        }
    }

    int total_count  = count;
    int count_buffer = count;

    if(id == 0) {
        MPI_Status status;
        for (int i = 1; i < n_processes; i++) {
			MPI_Recv(&count_buffer, sizeof(int), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            total_count += count_buffer;
		}
    } else {
        MPI_Send(&count_buffer, sizeof(int), MPI_INT, 0, id, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    // Calculate π.
    return ((double)total_count / n_samples) * 4.0;
}

double monte_carlo_pi_reduce(int n_samples, int seed) {
    return 0.0;
}

int main(int argc, char* argv[]) {

    double start_time, elapsed_time, pi;
    int is_correct;

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_serial(SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Serial:   pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_parallel(argc, argv, SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Parallel: pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    start_time = omp_get_wtime();
    pi = monte_carlo_pi_reduce(SAMPLES, SEED);
    elapsed_time = omp_get_wtime() - start_time;

    is_correct = (fabs(PI - pi) < 1e-3);
    printf("Reduce:   pi = %.9lf time = %.3lf correct = %d\n", pi, elapsed_time, is_correct);

    // Experimentation Area - Hazard
    
    return 0;
}