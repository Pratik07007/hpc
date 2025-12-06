#include<stdio.h>
#define MAX 1000

void main()
{
	double average=0.0, A[MAX];
	int i;
	for (i=0; i<MAX; i++)
	{
		A[i]=(double)i;
	}
	#pragma omp parallel for
	for (i=0; i<MAX; i++)
	{
		average += A[i];
	}
	average = average/MAX;
	printf("Average = %lf\n", average);
}
