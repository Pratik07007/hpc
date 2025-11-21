#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int counter = 0;
int flag = 0;

int doSomething(int a) {
    printf("Counter value: %d, a: %d\n", counter, a);
    return counter + a;
}

void* thread_one(void* arg) {
    for (int i = 0; i < 5; i++) {
        while (flag != 0);
        counter++;
        int result = doSomething(10);
        printf("Thread 1: doSomething returned %d\n", result);
        flag = 1; 
    }
    return NULL;
}

void* thread_two(void* arg) {
    for (int i = 0; i < 5; i++) {
        while (flag != 1);
        counter++;
        int result = doSomething(20);
        printf("Thread 2: doSomething returned %d\n", result);
        flag = 2;
        flag = 0;
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, thread_one, NULL);
    pthread_create(&t2, NULL, thread_two, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Final counter: %d\n", counter);
    return 0;
}
