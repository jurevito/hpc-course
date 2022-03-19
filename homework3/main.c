#include <math.h>
#include <omp.h>
#include <stdio.h>

#define TOL 1e-6

double f(double x) {
    return sin(x * x);
}

double quad(double (*f)(double), double a, double b, double tol) {
    double result = 0;

    double h = b - a;               // Interval length.
    double middle = (a + b) / 2;    // Middle point.

    // Compute both integral approximations.
    double quad_coarse = h * (f(a) + f(b)) / 2.0;
    double quad_fine = h / 2 * (f(a) + f(middle)) / 2.0 + h / 2 * (f(middle) + f(b)) / 2.0;
    double eps = fabs(quad_coarse - quad_fine);

    // If not precise enough split the interval on two parts.
    if (eps > tol) {
        double quad_a = quad(f, a, middle, tol / 2);
        double quad_b = quad(f, middle, b, tol / 2);
        result = quad_a + quad_b;
    } else {
        result = quad_fine;
    }

    return result;
}

double quad_parallel(double (*f)(double), double a, double b, double tol) {
    int n_intervals = 50000;
    double eps = (b - a) / n_intervals;
    double result = 0;

    #pragma omp parallel for schedule(guided, 4)
    for (int i = 0; i < n_intervals; i++) {
        double new_a = a + i * eps;
        double new_b = new_a + eps;

        #pragma omp atomic
        result += quad(f, new_a, new_b, tol);
    }

    return result;
}

int main(int argc, char* argv[]) {

    // Serial implementation.
    double start_time = omp_get_wtime();
    double result = quad(f, 0, 100, TOL);
    double elapsed_time = omp_get_wtime() - start_time;
    printf("(serial)   result: %lf time: %.3lf\n", result, elapsed_time);

    // Parallel implementation.
    start_time = omp_get_wtime();
    result = quad_parallel(f, 0, 100, TOL);
    elapsed_time = omp_get_wtime() - start_time;
    printf("(parallel) result: %lf time: %.3lf\n", result, elapsed_time);

    return 0;
}
