#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int numThreads = 10;
unsigned long lines = 0;

struct targs_t {
  long start;
  int *A, *B, *C;
  unsigned long lines;
  int numThreads;
};

__global__
void quad(void *p, double *d_X1, double *d_X2)
{
  struct targs_t ptargs =((struct targs_t *)p)[threadIdx.x];
  long end;
  unsigned long lines = ptargs.lines;
  int numThreads = ptargs.numThreads;
  long chunkSize = lines/numThreads;
  
  end = ptargs.start+chunkSize;
  if(end > lines)
    end = lines;
  
  for(long i=ptargs.start; i < end; i++){
    double a,b,c;
    double x1 = 0.0, x2 = 0.0;
    a = ptargs.A[i];
    b = ptargs.B[i];
    c = ptargs.C[i];
    x1 = ((-b) + sqrt(b*b - 4*a*c))/(2*a);
    x2 = ((-b) - sqrt(b*b - 4*a*c))/(2*a);
    d_X1[i] = x1;
    d_X2[i] = x2;
  }
}

int main()
{
  FILE *fp = NULL;
  int a,b,c;
  int *A, *B, *C;
  int *d_A, *d_B, *d_C;
  long line=0;
  struct targs_t *targs;
  double *X1, *X2;
  double *d_X1, *d_X2;
  
  
  fp = fopen("QuadData.txt","r");
  
  if(fp == NULL){
    printf("Error opening QuadData.txt\n");
    exit(-1);
  }
  
  while(!feof(fp)){
    fscanf(fp, "%d,%d,%d\n",&a,&b,&c);
    //printf("%d,%d,%d\n",a,b,c);
    lines++;
  }
  fclose(fp);  
  printf("lines = %ld\n",lines);
  A = (int *)malloc(lines * sizeof(int));
  B = (int *)malloc(lines * sizeof(int));
  C = (int *)malloc(lines * sizeof(int));
  cudaMalloc(&d_A, lines * sizeof(int));
  cudaMalloc(&d_B, lines * sizeof(int));
  cudaMalloc(&d_C, lines * sizeof(int));
  X1 = (double *)malloc(lines * sizeof(double));
  X2 = (double *)malloc(lines * sizeof(double));
  cudaMalloc(&d_X1, lines * sizeof(double));
  cudaMalloc(&d_X2, lines * sizeof(double));
  
  targs = (struct targs_t *)malloc(numThreads * sizeof(struct targs_t));  
  fp = fopen("QuadData.txt","r");
  
  while(!feof(fp)){
    fscanf(fp, "%d,%d,%d\n",&a,&b,&c);
    A[line] = a;
    B[line] = b;
    C[line] = c;
    line++;
  }
  fclose(fp);  
  line--;
  printf("%d,%d,%d\n",A[line],B[line],C[line]);
  cudaMemcpy(d_A, A, lines*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, lines*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, lines*sizeof(int), cudaMemcpyHostToDevice);
  
  for(int i=0; i < numThreads; i++){
    targs[i].start = i *(lines/numThreads);
    targs[i].A = A;
    targs[i].B = B;
    targs[i].C = C;
    targs[i].lines = lines;
    targs[i].numThreads = numThreads;
  }    
  quad<<<1,numThreads >>>(targs, d_X1, d_X2);
  cudaMemcpy(X1, d_X1, lines*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(X2, d_X2, lines*sizeof(double), cudaMemcpyDeviceToHost);
  for(int i = 0; i < lines; i++){
    printf("X1[%d] = %lf, X2[%d] = %lf\n", i, X1[i], i, X2[i]);
  }
  free(A);
  free(B);
  free(C);
  free(X1);
  free(X2);
  cudaFree(d_A);  
  cudaFree(d_B);  
  cudaFree(d_C);  
  cudaFree(d_X1);  
  cudaFree(d_X2);  
  free(targs);
  return 0;
}
