#include <stdio.h>

#define N 3

int main() {
    int A[N][N], B[N][N], C[N][N];

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            scanf("%d", &A[r][c]);

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            scanf("%d", &B[r][c]);

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[r][k] * B[k][c];
            C[r][c] = sum;
        }

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++)
            printf("%d ", C[r][c]);
        printf("\n");
    }

    return 0;
}
