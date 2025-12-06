#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel sections
    {
        #pragma omp section
        printf("A\n");
        #pragma omp section
        printf("B\n");
    }
}
