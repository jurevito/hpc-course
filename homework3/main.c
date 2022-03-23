#include <math.h>
#include <omp.h>
#include <stdio.h>

/*
tol <= 1e-6
scheduling: guided
n. subintervals: 93000
parallel result: 0.63142347
serial result:   0.63141792
┌────────────────┬──────────┬───────────┬───────────┬───────────┬───────────┐
│ n. threads     │ 1        │ 4         │ 8         │ 16        │ 32        │
├────────────────┼──────────┼───────────┼───────────┼───────────┼───────────┤
│ parallel time  │ 0.034    │ 0.009     │ 0.005     │ 0.003     │ 0.003     │
│ serial time    │ 13.188   │ 13.188    │ 13.188    │ 13.188    │ 13.188    │
│ speed up       │ 387.88x  │ 1465.33x  │ 2637.60x  │ 4396.00x  │ 4396.00x  │
└────────────────┴──────────┴───────────┴───────────┴───────────┴───────────┘

tol <= 1e-8
scheduling: guided
n. subintervals: 93000
parallel result: 0.63141797
serial result:   0.63141792
┌────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ n. threads     │ 1        │ 4        │ 8        │ 16       │ 32       │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ parallel time  │ 0.398    │ 0.100    │ 0.049    │ 0.025    │ 0.016    │
│ serial time    │ 128.453  │ 128.453  │ 128.453  │ 128.453  │ 128.453  │
│ speed up       │ 322.75x  │ 1284.53x │ 2621.49x │ 5138.12x │ 8028.31x │
└────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
*/

#define TOL 1e-8

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

    // Heuristic for splitting the interval into subintervals.
    int n_intervals = 1000 * (((b - a)*(b - a)) / 120 + 10);
    double h = (b - a) / n_intervals;
    double result = 0;

    #pragma omp parallel for schedule(guided) reduction(+:result)
    for (int i = 0; i < n_intervals; i++) {
        double new_a = a + i * h;
        double new_b = new_a + h;

        result += quad(f, new_a, new_b, tol);
    }

    return result;
}

int main(int argc, char* argv[]) {

    // Parallel implementation.
    double start_time = omp_get_wtime();
    double result = quad_parallel(f, 0, 100, TOL);
    double elapsed_time = omp_get_wtime() - start_time;
    printf("parallel result: %.8lf time: %.3lf\n", result, elapsed_time);

    // Serial implementation.
    start_time = omp_get_wtime();
    result = quad(f, 0, 100, TOL);
    elapsed_time = omp_get_wtime() - start_time;
    printf("serial result: %.8lf time: %.3lf\n", result, elapsed_time);

    return 0;
}
