#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#define i 4
#define j 3
//kernel function decleatation
__global__ void matadd(int *x,int *y, int *z)
{
//expression for matrix addition using gridDim and blockIndex
    int index=gridDim.x * blockIdx.y +blockIdx.x;
    z[index]=x[index]+y[index];
}
int main()
{
    int h_a[i][j];
    int h_b[i][j];
    int h_c[i][j];

    int *d_a,*d_b,*d_c;
    int r,c;  //r=row and c=column
//first matrix
printf("\nFirst matrix:\n");
    for(r=0;r<i; r++)
    {
        for(c=0;c<j;c++)
            {
	h_a[r][c]=r+c;
	printf("%d\t", h_a[r][c]);	
            }
	printf("\n");
    }
printf("\nSecond matrix:\n");
      for(r=0;r<i; r++)
    {
        for(c=0;c<j;c++)
            {
	h_b[r][c]=i*j+r;
	printf("%d\t", h_b[r][c]);
            }
	printf("\n");	
    }
   //memory allocation on device     
    cudaMalloc(&d_a,i*j*sizeof(int));
    cudaMalloc(&d_b,i*j*sizeof(int));
    cudaMalloc(&d_c,i*j*sizeof(int));
//copy matrices from host to device
 cudaMemcpy(d_a,h_a,i*j*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(d_b,h_b,i*j*sizeof(int),cudaMemcpyHostToDevice);
//kernel launch
 matadd<<<dim3(j,i),1>>>(d_a,d_b,d_c);
//copy result matrix from defice to host
 cudaMemcpy(h_c,d_c,i*j*sizeof(int),cudaMemcpyDeviceToHost);
//display result
    printf("\nSum of two matrices:\n");
    for(r=0;r<i;r++)
    {
        for(c=0;c<j;c++)
        {
              printf("%d\t",h_c[r][c]);
        }
        printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}