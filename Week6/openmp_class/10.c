#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp barrier
        printf("After barrier %d\n", omp_get_thread_num());
    }
}
