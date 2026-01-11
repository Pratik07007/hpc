#include <stdio.h>


__global__ void helloFromGPU() {
    
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}   