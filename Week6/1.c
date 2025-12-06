#include <stdio.h>
#include <omp.h>

void hello() {
    printf("Hello World from thread %d\n", omp_get_thread_num());
}

int main() {
    #pragma omp parallel num_threads(10)
    {
        hello();
    }
    return 0;
}


