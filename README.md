# High Performance Computing
Homework assignments for High Performance Computing course.
1. Basics of bash scripting and SLURM.
2. Integration using OpenMP.
3. Adaptive integration using OpenMP.
4. Sieve of Sieve of Eratosthenes using OpenMP.
5. RGB histogram from image using OpenCL.
6. Histogram equalization using OpenCL.
7. Matrix and vector multiplication using OpenCL.

### Code Linting
- Download formater using `npm clang-format`.
- Lint code: `clang-format -i -style=WebKit *.c *.h`.

### SSH HPC Cluster commands
- Connect to IJS cluster: `ssh js4204@nsc-login1.ijs.si`.
- Copy file to cluster: `scp .\main.c js4204@nsc-login1.ijs.si:./`.
- Download file from cluster `scp js4204@nsc-login1.ijs.si:./file.txt .\`.


### Bash Scripting
```bash
cat output_{0..4..1}.txt > output.txt && rm output_{0..4..1}.txt
```
