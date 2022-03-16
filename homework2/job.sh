#!/bin/sh
#SBATCH --job-name=integral
#SBATCH --time=00:02:00
#SBATCH --constraint=AMD
#SBATCH --reservation=fri
#SBATCH --cpus-per-task=32
#SBATCH --output=output_%a.txt
#SBATCH --array=0-4

THREADS=(1 4 8 16 32)

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=${THREADS[$SLURM_ARRAY_TASK_ID]}

module load GCC

gcc -O2 -fopenmp -lm -Wall main.c -o main

srun main 1000000000