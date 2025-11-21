#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define N 4

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;

int count = 0;
int generation = 0;

void barrier()
{
    pthread_mutex_lock(&m);

    int gen = generation;
    count++;

    if (count == N) {
        generation++;
        count = 0;
        pthread_cond_broadcast(&c);
    } else {
        while (gen == generation)
            pthread_cond_wait(&c, &m);
    }

    pthread_mutex_unlock(&m);
}

void* work(void* a)
{
    printf("T%d reached barrier 1\n", (int)(long)a);
    barrier();
    printf("T%d passed barrier 1\n", (int)(long)a);

    sleep(1);

    printf("T%d reached barrier 2\n", (int)(long)a);
    barrier();
    printf("T%d passed barrier 2\n", (int)(long)a);

    return NULL;
}

int main()
{
    pthread_t t[N];

    for (long i = 0; i < N; i++)
        pthread_create(&t[i], NULL, work, (void*)i);

    for (int i = 0; i < N; i++)
        pthread_join(t[i], NULL);

    return 0;
}
