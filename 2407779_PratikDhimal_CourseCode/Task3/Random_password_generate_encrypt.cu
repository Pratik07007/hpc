#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TOTAL_COMBOS 67600
#define ENC_LEN 11

// gpu error check
void check_gpu(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("gpu error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

__device__ void gpu_encrypt(char* out, int idx) {
    char raw[5];
    raw[0] = 'a' + (idx / 2600);
    idx %= 2600;
    raw[1] = 'a' + (idx / 100);
    idx %= 100;
    raw[2] = '0' + (idx / 10);
    raw[3] = '0' + (idx % 10);

    out[0] = raw[0] + 2;
    out[1] = raw[0] - 2;
    out[2] = raw[0] + 1;
    out[3] = raw[1] + 3;
    out[4] = raw[1] - 3;
    out[5] = raw[1] - 1;
    out[6] = raw[2] + 2;
    out[7] = raw[2] - 2;
    out[8] = raw[3] + 4;
    out[9] = raw[3] - 4;
    out[10] = '\0';

    // wrap chars
    for(int i=0; i<10; i++) {
        if(i < 6) {
            while(out[i] > 'z') out[i] -= 26;
            while(out[i] < 'a') out[i] += 26;
        } else {
            while(out[i] > '9') out[i] -= 10;
            while(out[i] < '0') out[i] += 10;
        }
    }
}

// kernel: generate cuda_hashes table
__global__ void gen_table(char* buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= TOTAL_COMBOS) return;

    // pointer to where this string goes in memory
    char* my_slot = &buffer[idx * ENC_LEN];
    gpu_encrypt(my_slot, idx);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <output_file>\n", argv[0]);
        return 1;
    }

    const char* output_file = argv[1];

    // file out
    FILE* fp = fopen(output_file, "w");
    if(!fp) {
        printf("Failed to open file: %s\n", output_file);
        return 1;
    }

    // allocate gpu memory
    char* d_buffer;
    size_t size = TOTAL_COMBOS * ENC_LEN;
    check_gpu(cudaMalloc(&d_buffer, size));

    // Dynamic launch configuration based on hardware
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Use half of max threads for safety
    int threads = prop.maxThreadsPerBlock / 2;
    // Align to 32 (warp size)
    threads = (threads / 32) * 32;
    if (threads == 0) threads = 32;

    int blocks = (TOTAL_COMBOS + threads - 1) / threads;

    printf("Kernel config: %d blocks, %d threads\n", blocks, threads);

    // launch kernel
    gen_table<<<blocks, threads>>>(d_buffer);
    check_gpu(cudaDeviceSynchronize());

    // copy back
    char* h_buffer = (char*)malloc(size);
    check_gpu(cudaMemcpy(h_buffer, d_buffer, size, cudaMemcpyDeviceToHost));

    // write to file
    for(int i=0; i<TOTAL_COMBOS; i++) {
        fprintf(fp, "%s\n", &h_buffer[i * ENC_LEN]);
    }

    fclose(fp);
    free(h_buffer);
    cudaFree(d_buffer);

    printf("generated %d encrypted combos to %s\n", TOTAL_COMBOS, output_file);
    return 0;
}