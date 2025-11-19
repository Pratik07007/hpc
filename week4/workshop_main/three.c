#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>

int is_prime(int n) { /* same */ }

typedef struct {
    int start, end, id;
} Range;

void *count_primes(void *arg) {
    Range *r = (Range *)arg;
    int count = 0;
    for (int i = r->start; i < r->end; i++)
        if (is_prime(i)) count++;
    int *result = malloc(sizeof(int));
    *result = count;
    pthread_exit(result);
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
        ranges[i].id = i;
        pthread_create(&threads[i], NULL, count_primes, &ranges[i]);
    }
    int total = 0;
    for (int i = 0; i < n; i++) {
        void *res;
        pthread_join(threads[i], &res);
        int count = *(int *)res;
        printf("Thread %d found %d primes\n", i, count);
        total += count;
        free(res);
    }
    printf("Total primes: %d\n", total);
    return 0;
}
