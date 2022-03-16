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

    #pragma omp parallel
    {
        #pragma omp for reduction(+:sum)
        for (int i = 1; i < n_intervals-1; i++) {
            double x = a + eps * i;
            sum += f(x);
        }
    }

    return (b - a)/n_intervals * (sum + (f(a)+f(b))/2);
}

int main(int argc, char** argv) {
    int n = (argc == 2) ? atoi(argv[1]): 1000000;
    int n_threads = omp_get_max_threads();
    
    double start_time = omp_get_wtime();
    double result = quad(f, 0, 100, n);
    double elapsed_time = omp_get_wtime() - start_time;

    printf("(%d threads) result: %.12lf time: %.6lf seconds\n", n_threads, result, elapsed_time);
}