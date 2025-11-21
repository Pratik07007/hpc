#include <stdio.h>
#include <pthread.h>

int counter = 0;

static void counter_func() {
    for (int i = 0; i < 100000; i++) {

       counter++;
    }
}

void* thread_one(void* arg) {
    counter_func();
    return NULL;
}

void* thread_two(void* arg) {
    counter_func();
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, thread_one, NULL);
    pthread_create(&t2, NULL, thread_two, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("%d\n", counter);
    return 0;
}
