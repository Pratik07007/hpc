//Program to square 100 numbers using CUDA

//Host is the CPU
//Device is the GPU

#include <stdio.h>

__global__ void square(int * device_input, int * device_output){

int threadId = threadIdx.x;
//printf("%d\n", threadId);
int  num = device_input[threadId];
device_output[threadId] = num * num;
//printf("%f\n", device_output[threadId]);
}

int main(int argc, char ** argv){
    int arraySize = 10; //size of list i want to create to square
    int arraySizeMemory = arraySize * sizeof(int); //memory amount
    
    //CPU - HOST Variables
    int hostNumbers[10];
    int hostSQNumbers[10];
    
    //creating input values on the CPU
    for (int i = 0; i<arraySize; i++){
        hostNumbers[i] =i;
    }
    
    //create GPU variables
    int * deviceNumbers;
    int * deviceSQNumbers;
    //hostNumbers = (float *) malloc(arraySizeMemory);
    //allocate memory on the GPU using cudaMallo
    //cudaMalloc ( variableName, memorySize)
    cudaMalloc( (void**) &deviceNumbers, arraySizeMemory);
    cudaMalloc( (void**) &deviceSQNumbers, arraySizeMemory);
    
    //transfer memory from CPU to GPU - cudaMemcpy
    //cudaMemcpy(destinationVar, originVar, memorySize, transferDirection)
    cudaMemcpy(deviceNumbers, hostNumbers, arraySizeMemory, cudaMemcpyHostToDevice);
    
    square<<<1 , 10>>>(deviceNumbers, deviceSQNumbers);
    cudaThreadSynchronize();
    
    cudaMemcpy(hostSQNumbers, deviceSQNumbers, arraySizeMemory, cudaMemcpyDeviceToHost);
    
    //free(hostNumbers);
	//free(hostSQNumbers);
	
	cudaFree(deviceNumbers);
	cudaFree(deviceSQNumbers);
	for(int i = 0; i<arraySize; i++){
       printf("%d\n", hostSQNumbers[i]);
    }
    
}





