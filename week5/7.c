#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

int value = 0;
pthread_mutex_t lock;

void* reader(void* arg) {
    for (int i = 0; i < 50000; i++) {
        pthread_mutex_lock(&lock);
        int v = value;
        pthread_mutex_unlock(&lock);
        (void)v;
    }
    return NULL;
}

void* writer(void* arg) {
    for (int i = 0; i < 50000; i++) {
        pthread_mutex_lock(&lock);
        value++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t r[8], w[2];
    pthread_mutex_init(&lock, NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < 8; i++)
        pthread_create(&r[i], NULL, reader, NULL);

    for (int i = 0; i < 2; i++)
        pthread_create(&w[i], NULL, writer, NULL);

    for (int i = 0; i < 8; i++)
        pthread_join(r[i], NULL);

    for (int i = 0; i < 2; i++)
        pthread_join(w[i], NULL);

    gettimeofday(&end, NULL);

    long micro = (end.tv_sec - start.tv_sec) * 1000000L +
                 (end.tv_usec - start.tv_usec);

    double seconds = micro / 1000000.0;

    printf("Final value: %d\n", value);
    printf("Execution time (seconds): %.6f\n", seconds);

    pthread_mutex_destroy(&lock);
    return 0;
}

