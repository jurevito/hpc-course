#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*
┌───────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ n. sub. \ n. th.  │ 1         │ 4         │ 8         │ 16        │ 32        │
├───────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 1.000.000         │ 0.071833  │ 0.019925  │ 0.010820  │ 0.007682  │ 0.005804  │
│ 100.000.000       │ 6.819708  │ 1.942543  │ 1.017409  │ 0.522906  │ 0.261771  │
│ 1.000.000.000     │ 68.090162 │ 18.053683 │ 9.842864  │ 5.024249  │ 2.559534  │
└───────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
*/

double f(double x) {
    return sin(x * x);
}

double quad(double (*f)(double), double a, double b, int n_intervals) {

    double eps = (b - a) / n_intervals; // Interval width.
    double sum = 0;

    int n_threads;
    int id;
    int i_start;
    int i_end;

    double partial_sum = 0.0;
    #pragma omp parallel private(n_threads, id, i_start, i_end) firstprivate(partial_sum)
    {
        n_threads = omp_get_num_threads();
        id = omp_get_thread_num();

        // Find iteration interval for thread.
        i_start = id * n_intervals / n_threads;
        i_end = (id + 1) * n_intervals / n_threads;

        for (int i = i_start; i < i_end; i++) {
            double x1 = a + eps * i;
            double x2 = eps + x1;

            partial_sum += (f(x1) + f(x2)) / 2;
        }

        #pragma omp atomic
        sum += partial_sum;
    }

    return (b - a)/n_intervals * sum;
}

int main(int argc, char** argv) {
    int n = (argc == 2) ? atoi(argv[1]): 1000000;
    
    double start_time = omp_get_wtime();
    double result = quad(f, -3, 10, n);
    double elapsed_time = omp_get_wtime() - start_time;
    printf("result: %.12lf\ntime: %.6lf seconds\n", result, elapsed_time);
}