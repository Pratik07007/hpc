#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>

int is_prime(int n) { /* same as above */ }

typedef struct {
    int start, end, thread_id;
} Range;

void *find_primes(void *arg) {
    Range *r = (Range *)arg;
    for (int i = r->start; i < r->end; i++)
        if (is_prime(i))
            printf("Thread %d: %d\n", r->thread_id, i);
    return NULL;
}

int main() {
    int n, limit = 10000;
    printf("Enter number of threads: ");
    scanf("%d", &n);
    pthread_t threads[n];
    Range ranges[n];
    int chunk = limit / n;
    for (int i = 0; i < n; i++) {
        ranges[i].start = i * chunk + 1;
        ranges[i].end = (i == n-1) ? limit + 1 : (i + 1) * chunk + 1;
        ranges[i].thread_id = i;
        pthread_create(&threads[i], NULL, find_primes, &ranges[i]);
    }
    for (int i = 0; i < n; i++) pthread_join(threads[i], NULL);
    return 0;
}
