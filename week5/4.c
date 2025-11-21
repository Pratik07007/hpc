#include <stdio.h>
#include <pthread.h>
#include <unistd.h>   // for usleep()

#define N 3

int A[N][N] = {
    {1,2,3},
    {4,5,6},
    {7,8,9}
};

int B[N][N] = {
    {1,0,1},
    {0,1,0},
    {1,0,1}
};

int C[N][N];
int temp;

void* work(void* arg) {
    int i = *(int*)arg;
    int r = i / N;
    int c = i % N;

    temp = 0;

    for (int k = 0; k < N; k++) {
        temp += A[r][k] * B[k][c];
        //usleep(1000);
    }

    C[r][c] = temp; 
    return NULL;
}

int main() {
    pthread_t t[N*N];
    int idx[N*N];

    for (int i = 0; i < N*N; i++) {
        idx[i] = i;
        pthread_create(&t[i], NULL, work, &idx[i]);
    }

    for (int i = 0; i < N*N; i++)
        pthread_join(t[i], NULL);

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++)
            printf("%d ", C[r][c]);
        printf("\n");
    }

    return 0;
}

