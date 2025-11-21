#include <stdio.h>
#include <pthread.h>

int counter = 0;
pthread_mutex_t lock;

static void counter_func() {
    for (int i = 0; i < 100000; i++) {

        pthread_mutex_lock(&lock);   // critical section START
        counter++;
        pthread_mutex_unlock(&lock); // critical section END
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

    pthread_mutex_init(&lock, NULL);

    pthread_create(&t1, NULL, thread_one, NULL);
    pthread_create(&t2, NULL, thread_two, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    pthread_mutex_destroy(&lock);

    printf("%d\n", counter);
    return 0;
}
