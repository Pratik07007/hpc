#include <stdio.h>
#include <pthread.h>

#define N 3

int A[N][N], B[N][N], C[N][N];

void* work(void* arg) {
    int i = *(int*)arg;
    int r = i / N;
    int c = i % N;

    int sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[r][k] * B[k][c];

    C[r][c] = sum;
    return NULL;
}

int main() {
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            scanf("%d", &A[r][c]);

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            scanf("%d", &B[r][c]);

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
