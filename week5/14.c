#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int flag = 0;

void* thread_one(void* arg) {
    printf("Thread 1: waiting for flag to become 1...\n");
    while (flag == 0);
    printf("Thread 1: detected flag = 1\n");
    return NULL;
}

void* thread_two(void* arg) {
    printf("Thread 2: sleeping for 2 seconds...\n");
    sleep(2);
    flag = 1;
    printf("Thread 2: flag set to 1\n");
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, thread_one, NULL);
    pthread_create(&t2, NULL, thread_two, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
