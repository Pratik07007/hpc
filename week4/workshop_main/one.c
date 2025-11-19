#include <stdio.h>
#include <pthread.h>
#include <math.h>

int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    return 1;
}

void *find_primes(void *arg) {
    int id = *(int *)arg;
    int start = 1 + id * 3334;
    int end = (id == 2) ? 10001 : start + 3334;
    for (int i = start; i < end; i++)
        if (is_prime(i))
            printf("Thread %d: %d is prime\n", id, i);
    return NULL;
}

int main() {
    pthread_t t[3];
    int ids[3] = {0, 1, 2};
    for (int i = 0; i < 3; i++)
        pthread_create(&t[i], NULL, find_primes, &ids[i]);
    for (int i = 0; i < 3; i++)
        pthread_join(t[i], NULL);
    return 0;
}
