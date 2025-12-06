#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp master
        printf("Master: %d\n", omp_get_thread_num());
    }
}
