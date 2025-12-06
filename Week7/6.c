#include<stdio.h>
#include<omp.h>
#define M 4
#define N 3

int main()
{
    double A[M][N] = {
        {1,2,3},
        {4,5,6},
        {7,8,9},
        {1,1,1}
    };

    double B[N][M] = {
        {1,2,3,4},
        {5,6,7,8},
        {9,1,2,3}
    };

    double C[M][M];

    #pragma omp parallel for
    for (int i=0; i<M; i++)
        for (int j=0; j<M; j++)
        {
            C[i][j] = 0.0;
            for (int k=0; k<N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    #pragma parallel for
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<M;j++)
            printf("%lf ", C[i][j]);
        printf("\n");
    }
}
