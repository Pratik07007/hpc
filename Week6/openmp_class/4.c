#include <stdio.h>
#include <omp.h>

int main() {
    int s=0;
    #pragma omp parallel for reduction(+:s)
    for(int i=1;i<=100;i++) s+=i;
    printf("%d\n", s);
}
