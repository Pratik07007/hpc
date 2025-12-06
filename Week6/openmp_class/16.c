#include <stdio.h>
#include <omp.h>

int isprime(int n) {
    if(n<2) return 0;
    for(int i=2;i*i<=n;i++)
        if(n%i==0) return 0;
    return 1;
}

int main() {
    #pragma omp parallel for schedule(dynamic)
    for(int i=1;i<=100000;i++)
        if(isprime(i))
            printf("%d\n", i);
}
