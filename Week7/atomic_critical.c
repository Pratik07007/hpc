#include <stdio.h>
#include <omp.h>

int main() {
    long x = 0, y = 0;
    double t1, t2;

    x = 0;
    t1 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < 100000000; i++)
        #pragma omp atomic
        x++;
    t1 = omp_get_wtime() - t1;

    y = 0;
    t2 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < 100000000; i++)
    {
        #pragma omp critical
        {
            y++;
        }
    }
    t2 = omp_get_wtime() - t2;

    printf("Atomic result:   %ld   Time: %f sec\n", x, t1);
    printf("Critical result: %ld   Time: %f sec\n", y, t2);

    return 0;
}
