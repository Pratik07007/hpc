#include <stdio.h>

#define N 3

int main() {
    int A[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int B[N][N] = {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
    };

    int C[N][N];

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {

            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[r][k] * B[k][c];
            }
            C[r][c] = sum;
        }
    }

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            printf("%d ", C[r][c]);
        }
        printf("\n");
    }

    return 0;
}


// For C[0][0] (first row, first column)

// Take row 0 of A → {1, 2, 3}
// Take column 0 of B → {1, 0, 1}

// Multiply corresponding elements:
//A[0][0] * B[0][0];   // 1 * 1 = 1
//A[0][1] * B[1][0];   // 2 * 0 = 0
//A[0][2] * B[2][0];   // 3 * 1 = 3

// Add them:
//1 + 0 + 3;           // = 4

// Final result:
//C[0][0] = 4;
