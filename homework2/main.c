#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*
quad(sin(x*x), 0, 100) = 0.631417952426
┌───────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ n. sub. \ n. th.  │ 1         │ 4         │ 8         │ 16        │ 32        │
├───────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 1.000.000         │ 0.035814  │ 0.009134  │ 0.005028  │ 0.003835  │ 0.003676  │
│ 100.000.000       │ 3.537702  │ 0.861746  │ 0.430468  │ 0.241757  │ 0.129418  │
│ 1.000.000.000     │ 35.160688 │ 8.587556  │ 4.300317  │ 2.386688  │ 1.272567  │
└───────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
*/

double f(double x) {
    return sin(x * x);
}

double quad(double (*f)(double), double a, double b, int n_intervals) {

    double eps = (b - a) / n_intervals; // Interval width.
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n_intervals-1; i++) {
        double x = a + eps * i;
        sum += f(x);
    }

    return (b - a)/n_intervals * (sum + (f(a)+f(b))/2);
}

int main(int argc, char** argv) {
    int n_intervals = atoi(argv[1]);
    
    double start_time = omp_get_wtime();
    double result = quad(f, 0, 100, n_intervals);
    double elapsed_time = omp_get_wtime() - start_time;

    printf("result: %.12lf\ntime: %.6lf seconds\n", result, elapsed_time);
    return 0;
}