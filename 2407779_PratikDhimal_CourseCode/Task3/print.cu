#include <stdio.h>

__global__ void PrintOut(){

//printf("threadIdx = %d\n", threadIdx.x);

//gridDim - gives us dimensions of the grid regarding blocks
//blockDim - gives us dimensions of the blocks regarding threads
//blockIdx - gives current block index
//threadIdx - gives current thread index

/*int threadId = threadIdx.x
             + threadIdx.y * blockDim.x
             + threadIdx.z * blockDim.x * blockDim.y
             + blockIdx.x * blockDim.x * blockDim.y * blockDim.z
             + blockIdx.y * gridDim.x * blockDim.x * blockDim.y * blockDim.z
             + blockIdx.z * gridDim.x * gridDim.y * blockDim.x * blockDim.y * blockDim.z;*/

//printf("gridDim.x = %d gridDim.y = %d gridDim.z = %d blockDim.x = %d blockDim.y = %d blockDim.z = %d blockIdx.x = %d blockIdx.y = %d blockIdx.z = %d threadIdx.x = %d threadIdx.y = %d threadIdx.z = %d\n",gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

printf("blockIdx.x = %d blockIdx.y = %d blockIdx.z = %d threadIdx.x = %d threadIdx.y = %d threadIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

int threadId = blockDim.x * blockIdx.x + threadIdx.x; //how many threads in total in the x * current block + current thread

printf("UID = %d\n", threadId);

//printf("%d")

}

int main(int argc, char ** argv){

//dim3 allows to use 3 dimensional configuration for blocks and threads

//PrintOut<<< dim3(1,2,3), dim3(4,5,6) >>>();
PrintOut<<< dim3(2,1,1), dim3(20,1,1) >>>(); //dim3(1,1,1), 
cudaThreadSynchronize();

}
