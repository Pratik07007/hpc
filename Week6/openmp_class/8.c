#include <stdio.h>
#include <omp.h>

int main() {
    int a=5;
    #pragma omp parallel firstprivate(a)
    printf("%d\n", a+1);
}
