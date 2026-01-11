#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//pi/4 = sigma  ((-1)^k) / ( (2*k) + 1) 

//k     0 -> 999

//[ , , , , , , , , , , , , ........ 999]

__global__ void Leibniz(double * leibnizOutput){

	int k = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("%d\n", k);
	
	leibnizOutput[k] = (pow(-1,k)) / (2*k+1);
}

int main(int argc, char ** argv){

	//1000 threads
	int blocksX = 10;
	int threadsX = 500;
	
	//double because we need decimal places
	int memorySize = blocksX * threadsX * sizeof(double);
	
	double * gpuList;
	double * cpuList = (double*) malloc (memorySize);
	
	cudaMalloc( (void**) &gpuList, memorySize);
	
	Leibniz<<< blocksX , threadsX >>>(gpuList);
	cudaThreadSynchronize();
	
	cudaMemcpy(cpuList, gpuList, memorySize, cudaMemcpyDeviceToHost);
	
	double finalSum = 0;
	
	for (int i = 0; i < blocksX * threadsX; i++){
		finalSum = finalSum + cpuList[i];
	}
	
	printf("Pi = %lf\n", finalSum * 4);

}
