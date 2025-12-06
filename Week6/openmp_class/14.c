#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel for schedule(dynamic,2)
    for(int i=0;i<10;i++)
        printf("%d\n", i);
}
