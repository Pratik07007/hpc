#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define HASH_LEN 100
#define RES_LEN 5
#define TABLE_SIZE 67600 

__device__ int is_match(const char* s1, const char* s2) {
    for(int i=0; i<HASH_LEN; i++) {
        if(s1[i] != s2[i]) return 0;
        if(s1[i] == '\0') break;
    }
    return 1;
}

// convert index back to raw passwords ( as required for us)
__device__ void idx_to_raw(int idx, char* out) {
    out[0] = 'a' + (idx / 2600);
    idx %= 2600;
    out[1] = 'a' + (idx / 100);
    idx %= 100;
    out[2] = '0' + (idx / 10);
    out[3] = '0' + (idx % 10);
    out[4] = '\0';
}

// kernel: compare targets vs cuda_hashes table
__global__ void crack_hashes(char* targets, char* cuda_hashes, char* results, int n_targets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_targets) return;

    char* my_target = &targets[idx * HASH_LEN];
    char* my_result = &results[idx * RES_LEN];

    int found = 0;
    for(int k=0; k<TABLE_SIZE; k++) {
        char* table_entry = &cuda_hashes[k * HASH_LEN];
        
        if(is_match(my_target, table_entry)) {
            idx_to_raw(k, my_result);
            found = 1;
            break;
        }
    }

    if(!found) {
        my_result[0] = '?'; my_result[1] = '?';
        my_result[2] = '?'; my_result[3] = '?';
        my_result[4] = '\0';
    }
}

// helper to read file into flat array
int load_file(const char* fname, char** buffer, int* count) {
    FILE* fp = fopen(fname, "r");
    if(!fp) return 0;

    if(*count == 0) {
        char tmp[200];
        while(fgets(tmp, 200, fp)) (*count)++;
        rewind(fp);
    }

    *buffer = (char*)malloc((*count) * HASH_LEN);
    char line[200];
    int i = 0;

    while(fgets(line, 200, fp) && i < *count) {
        line[strcspn(line, "\r\n")] = 0;
        strcpy(&(*buffer)[i * HASH_LEN], line);
        i++;
    }

    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    if(argc != 4) {
        printf("Usage: %s <target_hashes_file> <cuda_hashes_file> <output_file>\n", argv[0]);
        return 1;
    }

    // Load all files from command line
    const char* target_file = argv[1];
    const char* cuda_hash_file = argv[2];
    const char* output_file = argv[3];

    char* h_targets = NULL;
    int n_targets = 0;
    if(!load_file(target_file, &h_targets, &n_targets)) {
        printf("Failed to load target hashes file\n");
        return 1;
    }
    printf("loaded %d targets\n", n_targets);

    char* h_cuda_hashes = NULL;
    int n_table = TABLE_SIZE;
    if(!load_file(cuda_hash_file, &h_cuda_hashes, &n_table)) {
        printf("Failed to load cuda hashes file\n");
        free(h_targets);
        return 1;
    }

    // Allocate memory inside the device
    char *d_targets, *d_cuda_hashes, *d_results;
    cudaMalloc(&d_targets, n_targets * HASH_LEN);
    cudaMalloc(&d_cuda_hashes, TABLE_SIZE * HASH_LEN);
    cudaMalloc(&d_results, n_targets * RES_LEN);

    // Copy into the device
    cudaMemcpy(d_targets, h_targets, n_targets * HASH_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_hashes, h_cuda_hashes, TABLE_SIZE * HASH_LEN, cudaMemcpyHostToDevice);

    // Dynamic launch configuration based on hardware
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int threads = prop.maxThreadsPerBlock / 2;
    threads = (threads / 32) * 32;
    if (threads == 0) threads = 32;

    int blocks = (n_targets + threads - 1) / threads;

    printf("Kernel config: %d blocks, %d threads\n", blocks, threads);

    // Launch th kernel
    crack_hashes<<<blocks, threads>>>(d_targets, d_cuda_hashes, d_results, n_targets);
    cudaDeviceSynchronize();

    char* h_results = (char*)malloc(n_targets * RES_LEN);
    cudaMemcpy(h_results, d_results, n_targets * RES_LEN, cudaMemcpyDeviceToHost);

    FILE* fp = fopen(output_file, "w");
    if(!fp) {
        printf("Failed to open output file\n");
        return 1;
    }

    for(int i=0; i<n_targets; i++) {
        fprintf(fp, "%s\n", &h_results[i * RES_LEN]);
    }
    fclose(fp);

    printf("Passwords successfully cracked\n");

    free(h_targets);
    free(h_cuda_hashes);
    free(h_results);
    cudaFree(d_targets);
    cudaFree(d_cuda_hashes);
    cudaFree(d_results);

    return 0;
}