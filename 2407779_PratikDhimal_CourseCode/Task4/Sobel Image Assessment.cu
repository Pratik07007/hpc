#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h"

// CUDA KERNEL 1: RGB to Grayscale Conversion on GPU
__global__ void rgbToGrayscale(
    const unsigned char* rgba,
    unsigned char* gray,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int rgba_idx = idx * 4;
    
    unsigned char r = rgba[rgba_idx + 0];
    unsigned char g = rgba[rgba_idx + 1];
    unsigned char b = rgba[rgba_idx + 2];
    
    // Grayscale conversion formula based on lumionicity of human eye(not the standard average method)
    gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

// CUDA KERNEL 2: Sobel Edge Detection on GPU    
__global__ void sobelEdgeDetection(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int x = idx % width;
    int y = idx / width;

    
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}    
    };

    int sumX = 0;  
    int sumY = 0;  

    
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            // Calculate neighbor pixel coordinates
            int px = x + kx;
            int py = y + ky;

            // Zero padding: assume pixels outside image boundaries are 0
            unsigned char pixel = 0;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                pixel = input[py * width + px];
            }

            //multiply pixel value by kernel value and accumulate
            sumX += pixel * Gx[ky + 1][kx + 1];
            sumY += pixel * Gy[ky + 1][kx + 1];
        }
    }

    
    int magnitude = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
    
    // Clamp to valid pixel range [0, 255]
    if (magnitude > 255) magnitude = 255;

    output[idx] = (unsigned char)magnitude;
}


// CUDA KERNEL 3: Convert Grayscale back to RGBA for output

__global__ void grayToRGBA(
    const unsigned char* gray,
    unsigned char* rgba,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    unsigned char v = gray[idx];
    rgba[idx * 4 + 0] = v;  // R
    rgba[idx * 4 + 1] = v;  // G
    rgba[idx * 4 + 2] = v;  // B
    rgba[idx * 4 + 3] = 255;  // A (fully opaque)
}


int main() {
    unsigned char* h_rgba = NULL;
    unsigned width, height;

    printf("Loading image from file...\n");
    unsigned error = lodepng_decode32_file(&h_rgba, &width, &height, "hck.png");
    if (error) {
        printf("ERROR: Failed to load image - %s\n", lodepng_error_text(error));
        return 1;
    }
    printf("SUCCESS: Image loaded - %u x %u pixels\n", width, height);

    size_t pixels = width * height;
    size_t rgba_size = pixels * 4;  
    size_t gray_size = pixels;   // More specifically this coudl be called intensity value ( 1 value per pixel )     

    
    unsigned char* h_out_rgba = (unsigned char*)malloc(rgba_size);
    if (!h_out_rgba) {
        printf("ERROR: Failed to allocate host memory\n");
        free(h_rgba);
        return 1;
    }

    printf("\nAllocating GPU memory...\n");
    unsigned char *d_rgba, *d_gray, *d_edges, *d_out_rgba; //memoryAllocation
    
    cudaError_t err;
    
    err = cudaMalloc(&d_rgba, rgba_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_rgba - %s\n", cudaGetErrorString(err));
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    err = cudaMalloc(&d_gray, gray_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_gray - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    err = cudaMalloc(&d_edges, gray_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_edges - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    err = cudaMalloc(&d_out_rgba, rgba_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_out_rgba - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    printf("SUCCESS: GPU memory allocated\n");
    printf("  - Input RGBA: %zu bytes\n", rgba_size);
    printf("  - Grayscale: %zu bytes\n", gray_size);
    printf("  - Edges: %zu bytes\n", gray_size);
    printf("  - Output RGBA: %zu bytes\n", rgba_size);
    printf("\nTransferring data to GPU...\n");

    err = cudaMemcpy(d_rgba, h_rgba, rgba_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMemcpy H2D failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    printf("SUCCESS: Data transferred to GPU\n");

    
    printf("\nExecuting edge detection on GPU...\n");
    
    
    int blockSize = 256;  // 256 threads per block (1D) thsi is doen to to make a perfect multiple os 32 which is a single warp (32*8=256)
    int gridSize = (pixels + blockSize - 1) / blockSize;
    
    printf("  Grid: %d blocks\n", gridSize);
    printf("  Block: %d threads\n", blockSize);

    // Kernel 1: Converting RGB to Grayscale Intensity
    printf("\n  Step 1: Converting to grayscale...\n");
    rgbToGrayscale<<<gridSize, blockSize>>>(d_rgba, d_gray, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: rgbToGrayscale kernel launch failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: cudaDeviceSynchronize after grayscale failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    printf("  SUCCESS: Grayscale conversion complete\n");

    // Kernel 2: Apply Sobel Edge Detection on the Grayscale Intensity 
    printf("  Step 2: Applying Sobel edge detection...\n");
    sobelEdgeDetection<<<gridSize, blockSize>>>(d_gray, d_edges, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: sobelEdgeDetection kernel launch failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: cudaDeviceSynchronize after Sobel failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    printf("  SUCCESS: Edge detection complete\n");

    // Kernel 3: Convert back to RGBA for output 
    printf("  Step 3: Converting to RGBA for output...\n");
    grayToRGBA<<<gridSize, blockSize>>>(d_edges, d_out_rgba, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: grayToRGBA kernel launch failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: cudaDeviceSynchronize after RGBA conversion failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    printf("  SUCCESS: RGBA conversion complete\n");
    printf("SUCCESS: All GPU processing complete\n");

    
    printf("\nTransferring result from device to host...\n");
    err = cudaMemcpy(h_out_rgba, d_out_rgba, rgba_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMemcpy D2H failed - %s\n", cudaGetErrorString(err));
        cudaFree(d_rgba);
        cudaFree(d_gray);
        cudaFree(d_edges);
        cudaFree(d_out_rgba);
        free(h_rgba);
        free(h_out_rgba);
        return 1;
    }
    printf("SUCCESS: Result transferred to host\n");

    
    printf("\nSaving output image...\n");
    error = lodepng_encode32_file("sobel-final-image.png", h_out_rgba, width, height);
    if (error) {
        printf("ERROR: Failed to save image - %s\n", lodepng_error_text(error));
    } else {
        printf("SUCCESS: Output saved to 'sobel-final-image.png'\n");
    }

    
    printf("\nCleaning up...\n");
    
    // Free GPU memory
    cudaFree(d_rgba);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaFree(d_out_rgba);
    printf("  - GPU memory freed\n");
    
    // Free CPU memory
    free(h_rgba);
    free(h_out_rgba);
    printf("  - CPU memory freed\n");

    printf("\n=== Sobel Edge Detection Complete ===\n");
    return 0;
}