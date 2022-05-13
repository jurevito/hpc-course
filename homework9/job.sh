#!/bin/bash

srun --reservation=fri mpicc -O2 -lm -fopenmp main.c -o main

threads=( 1 4 8 32 )

touch output.txt & > output.txt

for i in "${threads[@]}" 
do
    printf "Number of Threads = $i Number of Nodes = 1\n" >> output.txt
    srun --mpi=pmix --ntasks=$i --nodes=1 --reservation=fri ./main >> output.txt

    if [ $i -gt 1 ] 
    then
        printf "Number of Threads = $i Number of Nodes = 2\n" >> output.txt
        srun --mpi=pmix --ntasks=$i --nodes=2 --reservation=fri ./main >> output.txt
    fi

    printf "=========================\n" >> output.txt
done