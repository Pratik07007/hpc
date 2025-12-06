#include<stdio.h>
#include<omp.h>
#define M 4
#define N 3

void main()
{
	double A[M][N], B[N][M], C[M][M];
	#pragma omp parallel for
	for (int i=0; i<M; i++)
	{
		for (int j=0; j<M; j++)
		{
			C[i][j] = 0.00;
			for (int k=0; k<N; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}
