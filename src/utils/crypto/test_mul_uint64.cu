#include <iostream>
#include <cuda_runtime.h>

// Device function to multiply two uint64_t numbers
__device__ void mul_uint64(uint64_t a, uint64_t b, uint64_t &high, uint64_t &low) {
    uint64_t a_low = a & 0xFFFFFFFF;
    uint64_t a_high = a >> 32;
    uint64_t b_low = b & 0xFFFFFFFF;
    uint64_t b_high = b >> 32;

    uint64_t low_low = a_low * b_low;
    uint64_t low_high = a_low * b_high;
    uint64_t high_low = a_high * b_low;
    uint64_t high_high = a_high * b_high;

    uint64_t carry = ((low_low >> 32) + (low_high & 0xFFFFFFFF) + (high_low & 0xFFFFFFFF)) >> 32;

    low = (low_low & 0xFFFFFFFF) | ((low_high + high_low) << 32);

    high = high_high + (low_high >> 32) + (high_low >> 32) + carry;
}

// CUDA kernel to call the device function
__global__ void test_mul_uint64(uint64_t a, uint64_t b, uint64_t *d_high, uint64_t *d_low, uint64_t *d_low_low) {
    uint64_t high, low;
    mul_uint64(a, b, high, low);
    *d_high = high;
    *d_low = low;
}

int main() {
    // Test values
    uint64_t a = 3;
    uint64_t b = 44789000;

    // Host variables to store the results
    uint64_t h_high, h_low, h_low_low;

    // Device variables to store the results
    uint64_t *d_high, *d_low, *d_low_low;
    cudaMalloc((void**)&d_high, sizeof(uint64_t));
    cudaMalloc((void**)&d_low, sizeof(uint64_t));
    cudaMalloc((void**)&d_low_low, sizeof(uint64_t));

    // Launch the kernel
    test_mul_uint64<<<1, 1>>>(a, b, d_high, d_low, d_low_low);

    // Copy the results from device to host
    cudaMemcpy(&h_high, d_high, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_low, d_low, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_low_low, d_low_low, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "Result:" << std::endl;
    std::cout << "High: " << h_high << std::endl;
    std::cout << "Low: " << h_low << std::endl;
    std::cout << "Low_Low: " << h_low_low << std::endl;

    // Free device memory
    cudaFree(d_high);
    cudaFree(d_low);
    cudaFree(d_low_low);

    return 0;
}
