#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double f(double x) {
    return sin(x * x);
}

double integral(double (*f)(double), double a, double b, int n_intervals) {

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

int main() {
    printf("result: %.12lf\n", integral(f, 0, 13, 1000000));
}