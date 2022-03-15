#!/bin/sh
#SBATCH --job-name=integral
#SBATCH --time=00:00:20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output.txt

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=4

module load GCC

# compile
gcc -O2 -fopenmp -lm -Wall main.c -o main

# run
srun main 1000000