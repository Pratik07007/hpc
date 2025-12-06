#include <stdio.h>
#include <omp.h>

int main() {
    long r = 0, c = 0, a = 0;
    double tr, tc, ta;

    tr = omp_get_wtime();
    #pragma omp parallel for reduction(+:r)
    for(long i=0;i<100000000;i++)
        r += 1;
    tr = omp_get_wtime() - tr;

    tc = omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<100000000;i++) {
        #pragma omp critical
        c += 1;
    }
    tc = omp_get_wtime() - tc;

    ta = omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<100000000;i++) {
        #pragma omp atomic
        a += 1;
    }
    ta = omp_get_wtime() - ta;

    printf("Reduction: %ld  Time: %f sec\n", r, tr);
    printf("Critical:  %ld  Time: %f sec\n", c, tc);
    printf("Atomic:    %ld  Time: %f sec\n", a, ta);

    return 0;
}
