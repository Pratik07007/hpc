#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#define i 2
#define k 3
#define j 2
//kernel function declearation
__global__ void MatrixMulFunc(int *x,int *y, int *z)
{
    int m=blockIdx.x;
    int n=blockIdx.y;
//matrix multiplication expression 
   z[j*n+m]=0;
    for(int p=0;p<k;p++){
	 z[j*n+m]+=x[k*n+p]*y[j*p+m];
	}
   }
int main()
{
    int h_a[i][k];
    int h_b[k][j];
    int h_c[i][j];

    int *d_a,*d_b,*d_c;
    int r,c;   //r=row and c=column
printf("First matrix:\n");
    for(r=0;r<i; r++)
    {
        for(c=0;c<k;c++)
            {
	h_a[r][c]=r+c;
	printf("%d\t", h_a[r][c]);	
            }
	printf("\n");
    }
printf("\nSecond matrix:\n");
      for(r=0;r<k; r++)
    {
        for(c=0;c<j;c++)
            {
	h_b[r][c]=(r+c)*k;
	printf("%d\t", h_b[r][c]);
            }
	printf("\n");	
    }
//allocate memory to store matrices h_a[i][k],h_b[k][j],h_c[i][j] in device   
    cudaMalloc(&d_a,i*k*sizeof(int));
    cudaMalloc(&d_b,k*j*sizeof(int));
    cudaMalloc(&d_c,i*j*sizeof(int));
//copy matrices h_a[i][k],h_b[k][j] from host to device
 cudaMemcpy(d_a,h_a,i*k*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(d_b,h_b,k*j*sizeof(int),cudaMemcpyHostToDevice);
//kernel function invocation usinf 2D block having one thread 
    MatrixMulFunc<<<dim3(j,k),1>>>(d_a,d_b,d_c);
//copy result matrix from device to host
 cudaMemcpy(h_c,d_c,i*j*sizeof(int),cudaMemcpyDeviceToHost);
//to display result matirx
    printf("\nSum of two matrices:\n");
    for(r=0;r<i;r++)
    {
        for(c=0;c<j;c++)
        {
              printf("%d\t",h_c[r][c]);
        }
        printf("\n");
    }
//de-allocate allocated memory of device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}