#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define HASH_LEN 100
#define PASSWORD_LEN 5
#define HASH_TABLE_SIZE 67600

__device__ int strings_equal(const char* a, const char* b) {
    for (int i = 0; i < HASH_LEN; i++) {
        if (a[i] != b[i]) return 0;
        if (a[i] == '\0') break;
    }
    return 1;
}

__device__ void index_to_password(int index, char* password) {
    password[0] = 'a' + (index / 2600);
    index %= 2600;
    password[1] = 'a' + (index / 100);
    index %= 100;
    password[2] = '0' + (index / 10);
    password[3] = '0' + (index % 10);
    password[4] = '\0';
}

__global__ void crack_passwords(
    char* target_hashes,
    char* precomputed_hashes,
    char* cracked_passwords,
    int target_count
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= target_count) return;

    char* current_target = &target_hashes[thread_id * HASH_LEN];
    char* output_password = &cracked_passwords[thread_id * PASSWORD_LEN];

    int match_found = 0;

    for (int table_index = 0; table_index < HASH_TABLE_SIZE; table_index++) {
        char* table_hash = &precomputed_hashes[table_index * HASH_LEN];

        if (strings_equal(current_target, table_hash)) {
            index_to_password(table_index, output_password);
            match_found = 1;
            break;
        }
    }

    if (!match_found) {
        output_password[0] = 'X';
        output_password[1] = 'X';
        output_password[3] = 'X';
        output_password[4] = '\0';
    }
}

int read_hash_file(const char* filename, char** buffer, int* line_count) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;

    if (*line_count == 0) {
        char temp[200];
        while (fgets(temp, 200, file)) (*line_count)++;
        rewind(file);
    }

    *buffer = (char*)malloc((*line_count) * HASH_LEN);

    char line[200];
    int i = 0;
    while (fgets(line, 200, file) && i < *line_count) {
        line[strcspn(line, "\r\n")] = 0;
        strcpy(&(*buffer)[i * HASH_LEN], line);
        i++;
    }

    fclose(file);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <target_hashes> <gpu_hash_bank> <output_file>\n", argv[0]);
        return 1;
    }

    const char* target_hash_file = argv[1];
    const char* hash_table_file = argv[2];
    const char* output_file = argv[3];

    char* host_targets = NULL;
    int target_count = 0;

    if (!read_hash_file(target_hash_file, &host_targets, &target_count)) {
        printf("Failed to load target hashes\n");
        return 1;
    }

    printf("Loaded %d target hashes\n", target_count);

    char* host_hash_table = NULL;
    int table_count = HASH_TABLE_SIZE;

    if (!read_hash_file(hash_table_file, &host_hash_table, &table_count)) {
        printf("Failed to load hash table\n");
        free(host_targets);
        return 1;
    }

    char *device_targets, *device_hash_table, *device_results;
    cudaMalloc(&device_targets, target_count * HASH_LEN);
    cudaMalloc(&device_hash_table, HASH_TABLE_SIZE * HASH_LEN);
    cudaMalloc(&device_results, target_count * PASSWORD_LEN);

    cudaMemcpy(device_targets, host_targets, target_count * HASH_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(device_hash_table, host_hash_table, HASH_TABLE_SIZE * HASH_LEN, cudaMemcpyHostToDevice);



    int blockSize = 256;  // 256 threads = 8 warps
    int gridSize  = (target_count + blockSize - 1) / blockSize;

    printf("Kernel launch: %d blocks, %d threads\n", gridSize, blockSize);



    crack_passwords<<<gridSize, blockSize>>>(
        device_targets,
        device_hash_table,
        device_results,
        target_count
    );

    cudaDeviceSynchronize();

    char* host_results = (char*)malloc(target_count * PASSWORD_LEN);
    cudaMemcpy(host_results, device_results, target_count * PASSWORD_LEN, cudaMemcpyDeviceToHost);

    FILE* output = fopen(output_file, "w");
    if (!output) {
        printf("Failed to open output file\n");
        return 1;
    }

    for (int i = 0; i < target_count; i++) {
        fprintf(output, "%s\n", &host_results[i * PASSWORD_LEN]);
    }

    fclose(output);

    printf("Password cracking completed successfully\n");

    free(host_targets);
    free(host_hash_table);
    free(host_results);
    cudaFree(device_targets);
    cudaFree(device_hash_table);
    cudaFree(device_results);

    return 0;
}
