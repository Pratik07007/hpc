#include <stdio.h>
#include <string.h>
#include <pthread.h>

void* thread_func(void* arg) {
    char* text = (char*)arg;
    char* saveptr;
    char* token = strtok_r(text, " ", &saveptr);

    while (token != NULL) {
        printf("%s\n", token);
        token = strtok_r(NULL, " ", &saveptr);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    char str1[] = "I am the best student of HCK";
    char str2[] = "Thread safe strtok_r example demo";

    pthread_create(&t1, NULL, thread_func, str1);
    pthread_create(&t2, NULL, thread_func, str2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
