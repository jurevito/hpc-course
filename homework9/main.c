#include "/usr/include/openmpi-x86_64/mpi.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEED 42
#define FIELD_SIZE 100000000

/*
module load OpenMPI
export OMPI_MCA_pml=ucx

┌─────────────┬───────────┬────────┬─────────┬────────┐
│ n. threads  │ n. nodes  │ start  │ middle  │ end    │
├─────────────┼───────────┼────────┼─────────┼────────┤
│ 1           │ 1         │ 0.042  │ 0.170   │ 0.299  │
│ 4           │ 1         │ 0.059  │ 0.076   │ 0.125  │
│ 8           │ 1         │ 0.068  │ 0.068   │ 0.101  │
│ 32          │ 1         │ 0.106  │ 0.105   │ 0.119  │
│ 4           │ 2         │ 0.061  │ 0.078   │ 0.182  │
│ 8           │ 2         │ 0.234  │ 0.116   │ 0.159  │
│ 32          │ 2         │ 0.459  │ 0.520   │ 0.418  │
└─────────────┴───────────┴────────┴─────────┴────────┘

Serial (start): 0.000
Serial (end): 0.118
*/

int find_intruder_serial(char* field, int size) {
    for (int i = 0; i < size; i++) {
        if (field[i] == 2) {
            return i;
        }
    }

    return -1;
}

int find_intruder(char* field, int size) {

    int id, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    int* send_counts = (int*)malloc(n_processes * sizeof(int));
    int* displacements = (int*)malloc(n_processes * sizeof(int));

    int sum = 0;
    int rem = size % n_processes;

    // Find indeces for spliting the vector between processes.
    for (int i = 0; i < n_processes; i++) {
        send_counts[i] = size / n_processes;
        if (rem > 0) {
            send_counts[i]++;
            rem--;
        }

        displacements[i] = sum;
        sum += send_counts[i];
    }

    char* recv_buffer = (char*)malloc(send_counts[id] * sizeof(char));
    MPI_Scatterv(field, send_counts, displacements, MPI_CHAR, recv_buffer, send_counts[id], MPI_CHAR, 0, MPI_COMM_WORLD);

    int intruder_found, intruder_index, n_swept;
    MPI_Request recv_request, send_request;

    MPI_Irecv(&intruder_index, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);

    for (int i = 0; i < send_counts[id]; i++) {
        // Process found the intruder.
        if (recv_buffer[i] == 2) {
            intruder_index = displacements[id] + i;
            n_swept = i;

            for (int j = 0; j < n_processes; j++) {
                MPI_Isend(&intruder_index, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &send_request);
            }
            break;
        }

        // Checking if any other process found the intruder.
        if (i % 1000 == 0) {
            MPI_Test(&recv_request, &intruder_found, MPI_STATUS_IGNORE);

            if (intruder_found) {
                n_swept = i;
                break;
            }
        }
    }

    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

    int* swept_buffer = (int*)malloc(n_processes * sizeof(int));
    MPI_Gather(&n_swept, 1, MPI_INT, swept_buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (id == 0) {
        for (int i = 0; i < n_processes; i++) {
            printf("(%d) swept %d fields.\n", i, swept_buffer[i]);
        }
    }

    return intruder_index;
}

int main(int argc, char* argv[]) {

    int provided_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_level);

    int id, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    char* field;
    if (id == 0) {
        srand(SEED);
        field = (char*)malloc(FIELD_SIZE * sizeof(char));

        for (int i = 0; i < FIELD_SIZE; i++) {
            field[i] = rand() % 2;
        }

        int section_size = FIELD_SIZE / n_processes;
        //field[0] = 2;            // Beginning
        //field[(n_processes / 2) * section_size + section_size / 2] = 2; // Middle
        field[FIELD_SIZE - 1] = 2; // End
    }

    double start_time, elapsed_time;
    int index;

    if (id == 0) {
        start_time = omp_get_wtime();
        index = find_intruder_serial(field, FIELD_SIZE);
        elapsed_time = omp_get_wtime() - start_time;
        printf("Serial: index = %d time = %.3lf\n", index, elapsed_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    start_time = omp_get_wtime();
    index = find_intruder(field, FIELD_SIZE);
    elapsed_time = omp_get_wtime() - start_time;
    if (id == 0) {
        printf("OpenMPI: index = %d time = %.3lf\n", index, elapsed_time);
    }

    MPI_Finalize();
    return 0;
}