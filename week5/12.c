#include <stdio.h>
#include <string.h>
#include <pthread.h>

void* thread_func(void* arg) {
    char* text = (char*)arg;
    char* token = strtok(text, " ");

    while (token != NULL) {
        printf("%s\n", token);
        for (volatile int i = 0; i < 1000000; i++);
        token = strtok(NULL, " ");
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    char str1[] = "I am the best student of HCK";
    char str2[] = "Thread unsafe strtok example demo";

    pthread_create(&t1, NULL, thread_func, str1);
    pthread_create(&t2, NULL, thread_func, str2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
