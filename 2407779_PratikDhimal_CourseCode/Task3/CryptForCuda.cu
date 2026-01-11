#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "sha512.h"

#define HASH_SIZE 64 // SHA-512 outputs 64 bytes
#define MAX_PASSWORDS 1000 // Max passwords to crack

// Structure to hold results
typedef struct {
    int found_count;
    char found_passwords[MAX_PASSWORDS][11]; // Store the intermediate 10-char password
} ResultStore;

// Helper to check errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Device function to compare two hashes
__device__ int compare_hashes(unsigned char *hash1, unsigned char *hash2) {
    for (int i = 0; i < HASH_SIZE; i++) {
        if (hash1[i] != hash2[i]) return 0;
    }
    return 1;
}

// Device function: The "Math Encryption" step
// Input: raw (4 chars, e.g., "aa99")
// Output: enc (10 chars + null)
__device__ void apply_math_encryption(char *raw, char *out) {
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

    for (int i = 0; i < 10; ++i) {
        if (i < 6) { // First 6 are letters
            while (out[i] > 'z') out[i] -= 26;
            while (out[i] < 'a') out[i] += 26;
        } else { // Last 4 are numbers
            while (out[i] > '9') out[i] -= 10;
            while (out[i] < '0') out[i] += 10;
        }
    }
}

// Kernel
__global__ void crack_passwords(unsigned char *target_hashes, int num_targets, ResultStore *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total combinations: 26 * 26 * 10 * 10 = 67600
    if (idx >= 67600) return;

    // 1. Generate Raw Password "aa00"
    int temp = idx;
    int d2 = temp % 10;
    temp /= 10;
    int d1 = temp % 10;
    temp /= 10;
    int c2 = temp % 26;
    temp /= 26;
    int c1 = temp; 

    char raw_password[5];
    raw_password[0] = 'a' + c1;
    raw_password[1] = 'a' + c2;
    raw_password[2] = '0' + d1;
    raw_password[3] = '0' + d2;
    raw_password[4] = '\0';

    // 2. Apply Math Encryption -> 10 char string
    char encrypted_math[12];
    apply_math_encryption(raw_password, encrypted_math);

    // 3. Apply SHA-512 Hashing
    unsigned char hash[HASH_SIZE];
    SHA512_CTX ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, (unsigned char*)encrypted_math, 10); // Length is 10
    sha512_final(&ctx, hash);

    // 4. Compare with Targets
    for (int i = 0; i < num_targets; i++) {
        unsigned char *target = &target_hashes[i * HASH_SIZE];
        if (compare_hashes(hash, target)) {
            int pos = atomicAdd(&(results->found_count), 1);
            if (pos < MAX_PASSWORDS) {
                // Store the intermediate (math-encrypted) password as requested
                // or the raw one. Let's store the text found in the file.
                for(int k=0; k<=10; k++) results->found_passwords[pos][k] = encrypted_math[k];
            }
        }
    }
}

// Host helper to convert hex string to bytes
void hex_to_bytes(const char *hex, unsigned char *bytes) {
    for (int i = 0; i < HASH_SIZE; i++) {
        sscanf(hex + 2*i, "%2hhx", &bytes[i]);
    }
}

int main() {
    // 1. Read Encrypted Passwords (these are SHA-512 hashes of the 10-char strings)
    FILE *fp = fopen("encrypted_passwords.txt", "r");
    if (!fp) {
        printf("Error: Could not open encrypted_passwords.txt.\n");
        return 1;
    }

    unsigned char *h_target_hashes = (unsigned char*)malloc(MAX_PASSWORDS * HASH_SIZE);
    int num_targets = 0;
    char line[200];

    while (fgets(line, sizeof(line), fp)) {
        if (num_targets >= MAX_PASSWORDS) break;
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) < 128) continue; 

        hex_to_bytes(line, &h_target_hashes[num_targets * HASH_SIZE]);
        num_targets++;
    }
    fclose(fp);

    printf("Loaded %d target hashes.\n", num_targets);

    // 2. Allocate Device Memory
    unsigned char *d_target_hashes;
    ResultStore *d_results;
    
    checkCudaError(cudaMalloc(&d_target_hashes, num_targets * HASH_SIZE), "Malloc Targets");
    checkCudaError(cudaMalloc(&d_results, sizeof(ResultStore)), "Malloc Results");

    checkCudaError(cudaMemcpy(d_target_hashes, h_target_hashes, num_targets * HASH_SIZE, cudaMemcpyHostToDevice), "Memcpy Targets");
    checkCudaError(cudaMemset(d_results, 0, sizeof(ResultStore)), "Memset Results");

    // 3. Launch Kernel
    int blockSize = 256;
    int numBlocks = (67600 + blockSize - 1) / blockSize;

    printf("Cracking...\n");
    crack_passwords<<<numBlocks, blockSize>>>(d_target_hashes, num_targets, d_results);
    
    checkCudaError(cudaDeviceSynchronize(), "Kernel Sync");

    // 4. Retrieve Results
    ResultStore h_results;
    checkCudaError(cudaMemcpy(&h_results, d_results, sizeof(ResultStore), cudaMemcpyDeviceToHost), "Memcpy Results");

    printf("Found %d matches.\n", h_results.found_count);

    FILE *out = fopen("decrypted_passwords.txt", "w");
    if (out) {
        for (int i = 0; i < h_results.found_count; i++) {
            fprintf(out, "%s\n", h_results.found_passwords[i]);
        }
        fclose(out);
        printf("Matches saved to decrypted_passwords.txt\n");
    }

    free(h_target_hashes);
    cudaFree(d_target_hashes);
    cudaFree(d_results);

    return 0;
}
