#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

sem_t printers;

void *print_document(void *arg) {
    int user = *(int *)arg;
    printf("User %d wants to print\n", user);
    sem_wait(&printers);
    printf(">>> User %d is printing...\n", user);
    sleep(2);  // printing time
    printf("<<< User %d finished printing\n", user);
    sem_post(&printers);
    return NULL;
}

int main() {
    sem_init(&printers, 0, 2);  // only 2 printers available
    pthread_t users[10];
    int ids[10] = {1,2,3,4,5,6,7,8,9,10};
    for (int i = 0; i < 10; i++)
        pthread_create(&users[i], NULL, print_document, &ids[i]);
    for (int i = 0; i < 10; i++)
        pthread_join(users[i], NULL);
    sem_destroy(&printers);
    return 0;
}
