#include <stdio.h>
#include <omp.h>

int main() {
    long sum1 = 0, sum2 = 0;
    double t1, t2;

    t1 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum1)
    for (long i = 0; i < 100000000; i++)
        sum1 += 1;
    t1 = omp_get_wtime() - t1;

    t2 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < 100000000; i++) {
        #pragma omp critical
        sum2 += 1;
    }
    t2 = omp_get_wtime() - t2;

    printf("Reduction: %ld  Time: %f sec\n", sum1, t1);
    printf("Critical:  %ld  Time: %f sec\n", sum2, t2);

    return 0;
}
