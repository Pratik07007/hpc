#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdatomic.h>

_Atomic int prime_count = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int is_prime(int n) { /* same */ }

void *find_and_count(void *arg) {
    int start = *(int *)arg;
    for (int i = start; i <= 10000; i += 8) {  // 8 threads max safe step
        if (is_prime(i)) {
            pthread_mutex_lock(&mutex);
            prime_count++;
            if (prime_count >= 5) {
                pthread_mutex_unlock(&mutex);
                return NULL;
            }
            printf("Found prime %d: %d\n", prime_count, i);
            pthread_mutex_unlock(&mutex);
        }
    }
    return NULL;
}

int main() {
    int n;
    printf("Enter number of threads: ");
    scanf("%d", &n);
    pthread_t threads[n];
    int starts[n];
    for (int i = 0; i < n; i++) {
        starts[i] = 2 + i;
        pthread_create(&threads[i], NULL, find_and_count, &starts[i]);
    }
    while (prime_count < 5) usleep(1000);  // wait until 5th found
    for (int i = 0; i < n; i++)
        pthread_cancel(threads[i]);
    for (int i = 0; i < n; i++)
        pthread_join(threads[i], NULL);
    printf("5th prime found, all threads cancelled. Total found: %d\n", prime_count);
    return 0;
}
