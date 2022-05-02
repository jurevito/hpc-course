compile:
	srun --reservation=fri mpicc -O2 -lm -fopenmp main.c -o main
run:
	srun --mpi=pmix --ntasks=1 --nodes=1 --reservation=fri ./main
format: 
	clang-format -i -style=file *.c
.PHONY: compile, run, format