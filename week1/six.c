#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void *threadFunc(void *arg)
{
    for (int i = 0; i < 10; i++)
    {
        printf("Thread ID %ld: i=%d\n", pthread_self(), i);
        usleep(1000);
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    int numThreads = atoi(argv[1]);

    pthread_t *threads = malloc(numThreads * sizeof(pthread_t));

    for (int i = 0; i < numThreads; i++)
    {
        if (pthread_create(&threads[i], NULL, threadFunc, NULL) != 0)
        {
            perror("Failed to create thread");
            return 1;
        }
    }

    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    return 0;
}

