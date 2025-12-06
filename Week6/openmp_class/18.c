#include <stdio.h>
#include <omp.h>

int isprime(int n) {
    if(n<2) return 0;
    for(int i=2;i*i<=n;i++)
        if(n%i==0) return 0;
    return 1;
}

int main() {
    int n;
    int count=0;
    scanf("%d",&n);

    #pragma omp parallel for reduction(+:count)
    for(int i=1;i<=n;i++)
        if(isprime(i)) count++;

    if(count>500) {
        omp_set_num_threads(2);
        #pragma omp parallel for schedule(dynamic)
        for(int i=1;i<=n;i++)
            if(isprime(i))
                printf("%d\n", i);
    } else {
        #pragma omp parallel for
        for(int i=1;i<=n;i++)
            if(isprime(i))
                printf("%d\n", i);
    }
}
