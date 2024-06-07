#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function to add elements of two arrays
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);

    // Allocate memory on the host
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define the number of blocks and threads
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy the result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result and print test cases
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error: Value mismatch at index " << i << std::endl;
            std::cerr << "Expected " << h_a[i] + h_b[i] << " but got " << h_c[i] << std::endl;
            success = false;
        }
        // Print test case
        std::cout << "h_a[" << i << "] + h_b[" << i << "] = " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    if (success) {
        std::cout << "CUDA addition was successful!" << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
