#include <stdio.h>
#include <omp.h>

int main() {
    omp_set_nested(1);
    #pragma omp parallel num_threads(2)
    {
        #pragma omp parallel num_threads(2)
        printf("%d\n", omp_get_thread_num());
    }
}
