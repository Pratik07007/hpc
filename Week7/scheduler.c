/*
static  = Equal fixed-size chunks assigned before execution. Lowest overhead.
dynamic = Threads grab new chunks when they finish. Good for uneven workloads.
guided  = Large chunks first, smaller later. Balances load with less overhead than dynamic.
auto    = Compiler/runtime decides the best scheduling automatically.
*/

#include <stdio.h>
#include <omp.h>

void run(const char *name, int schedule_type) {
    double t = omp_get_wtime();
    long sum = 0;

    switch(schedule_type) {
        case 0:
            #pragma omp parallel for schedule(static)
            for(long i=0;i<200000000;i++) sum += i;
            break;
        case 1:
            #pragma omp parallel for schedule(dynamic)
            for(long i=0;i<200000000;i++) sum += i;
            break;
        case 2:
            #pragma omp parallel for schedule(guided)
            for(long i=0;i<200000000;i++) sum += i;
            break;
        case 3:
            #pragma omp parallel for schedule(auto)
            for(long i=0;i<200000000;i++) sum += i;
            break;
    }

    t = omp_get_wtime() - t;
    printf("%s time: %f sec\n", name, t);
}

int main() {
    run("static", 0);
    run("dynamic", 1);
    run("guided", 2);
    run("auto", 3);
}
