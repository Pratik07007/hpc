#include <stdio.h>
#include <omp.h>

int isprime(int n) {
    if(n<2) return 0;
    for(int i=2;i*i<=n;i++)
        if(n%i==0) return 0;
    return 1;
}

void p1() {
    for(int i=1;i<=50000;i++)
        if(isprime(i))
            printf("%d\n", i);
}

void p2() {
    for(int i=50001;i<=100000;i++)
        if(isprime(i))
            printf("%d\n", i);
}

int main() {
    #pragma omp parallel sections
    {
        #pragma omp section
        p1();
        #pragma omp section
        p2();
    }
}
