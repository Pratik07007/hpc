#include <stdio.h>
#include <omp.h>

int main() {
    int sum = 0;
    #pragma omp parallel for
    for(int i = 1; i <= 100; i++) sum += i;

    printf("%d\n", sum);
}
