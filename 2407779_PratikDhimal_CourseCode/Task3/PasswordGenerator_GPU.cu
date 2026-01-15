#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TOTAL_COMBOS 67600
#define ENC_LEN 11

__global__ void create_password(char* buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_COMBOS) return;

    char raw[5];

    raw[0] = 'a' + (idx / 2600);
    idx %= 2600;
    raw[1] = 'a' + (idx / 100);
    idx %= 100;
    raw[2] = '0' + (idx / 10);
    raw[3] = '0' + (idx % 10);

    char* out = &buffer[(blockIdx.x * blockDim.x + threadIdx.x) * ENC_LEN];

    out[0]  = raw[0] + 2;
    out[1]  = raw[0] - 2;
    out[2]  = raw[0] + 1;
    out[3]  = raw[1] + 3;
    out[4]  = raw[1] - 3;
    out[5]  = raw[1] - 1;
    out[6]  = raw[2] + 2;
    out[7]  = raw[2] - 2;
    out[8]  = raw[3] + 4;
    out[9]  = raw[3] - 4;
    out[10] = '\0';

    for (int i = 0; i < 10; i++) {
        if (i < 6) {
            while (out[i] > 'z') out[i] -= 26;
            while (out[i] < 'a') out[i] += 26;
        } else {
            while (out[i] > '9') out[i] -= 10;
            while (out[i] < '0') out[i] += 10;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <output_file>\n", argv[0]);
        return 1;
    }

    FILE* fp = fopen(argv[1], "w");
    if (!fp) {
        printf("Failed to open file\n");
        return 1;
    }

    size_t size = TOTAL_COMBOS * ENC_LEN;

    char* d_buffer;
    if (cudaMalloc(&d_buffer, size) != cudaSuccess) {
        printf("cudaMalloc failed\n");
        return 1;
    }

    

    int blockSize = 256;  // 256 threads per block (8 warps)
    int gridSize  = (TOTAL_COMBOS + blockSize - 1) / blockSize;

    printf("Kernel config: %d blocks, %d threads\n", gridSize, blockSize);



    create_password<<<gridSize, blockSize>>>(d_buffer);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("Kernel execution failed\n");
        return 1;
    }

    char* h_buffer = (char*)malloc(size);
    if (!h_buffer) {
        printf("Host malloc failed\n");
        return 1;
    }

    if (cudaMemcpy(h_buffer, d_buffer, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Memcpy failed\n");
        return 1;
    }

    for (int i = 0; i < TOTAL_COMBOS; i++) {
        fprintf(fp, "%s\n", &h_buffer[i * ENC_LEN]);
    }

    fclose(fp);
    free(h_buffer);
    cudaFree(d_buffer);

    printf("generated %d encrypted combos\n", TOTAL_COMBOS);
    return 0;
}
