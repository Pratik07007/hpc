#include <stdio.h>
#include <omp.h>

int main() {
    int x=0;
    #pragma omp parallel shared(x)
    printf("%d\n", x);
}
