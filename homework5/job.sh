#!/bin/sh
#SBATCH --job-name=sieve
#SBATCH --time=00:05:00
#SBATCH --constraint=AMD
#SBATCH --reservation=fri
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

module load GCC
module load CUDA

gcc -O2 -lm -fopenmp -lOpenCL -Wall main.c -o main
srun -n1 main