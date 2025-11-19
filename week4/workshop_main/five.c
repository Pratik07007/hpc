#include <stdio.h>
#include <pthread.h>

typedef struct {
    int accountNumber;
    double balance;
    pthread_mutex_t lock;
} Account;

Account accounts[10];

void *withdraw(void *p) {
    int id = *(int *)p;
    double amount = 100.0;
    pthread_mutex_lock(&accounts[id].lock);
    accounts[id].balance -= amount;
    printf("Withdrew %lf from account %d, balance = %.2f\n", amount, id, accounts[id].balance);
    pthread_mutex_unlock(&accounts[id].lock);
    return NULL;
}

void *deposit(void *p) {
    int id = *(int *)p;
    double amount = 100.0;
    pthread_mutex_lock(&accounts[id].lock);
    accounts[id].balance += amount;
    printf("Deposited %lf to account %d, balance = %.2f\n", amount, id, accounts[id].balance);
    pthread_mutex_unlock(&accounts[id].lock);
    return NULL;
}

int main() {
    pthread_t threads[20];
    int ids[10];
    for (int i = 0; i < 10; i++) {
        accounts[i].accountNumber = i;
        accounts[i].balance = 1000.0;
        pthread_mutex_init(&accounts[i].lock, NULL);
        ids[i] = i;
    }
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, withdraw, &ids[i]);
        pthread_create(&threads[i+10], NULL, deposit, &ids[i]);
    }
    for (int i = 0; i < 20; i++) pthread_join(threads[i], NULL);
    for (int i = 0; i < 10; i++) {
        printf("Final Account %d balance = %.2f\n", i, accounts[i].balance);
        pthread_mutex_destroy(&accounts[i].lock);
    }
    return 0;
}
