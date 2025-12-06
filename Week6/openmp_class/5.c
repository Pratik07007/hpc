#include <stdio.h>
#include <omp.h>

int main() {
    int x=0;
    #pragma omp parallel
    {
        #pragma omp critical
        x++;
    }
    printf("%d\n", x);
}
