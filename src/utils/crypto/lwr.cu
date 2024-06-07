#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

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

__global__ void matrixMultiplyKernel(const uint64_t *A_high, const uint64_t *A_low, const uint64_t *B_high, const uint64_t *B_low, uint64_t *C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        uint64_t c_high = 0;
        uint64_t inter_high, inter_low;

        for (int k = 0; k < m; k++) {
            uint64_t a_low = A_low[row * m + k];
            uint64_t b_low = B_low[k * p + col];

            mul_uint64(a_low, b_low, inter_high, inter_low);
            c_high += inter_high;

            uint64_t b_high = B_high[k * p + col];

            mul_uint64(a_low, b_high, inter_high, inter_low);
            c_high += inter_low;

            uint64_t a_high = A_high[row * m + k];

            mul_uint64(a_high, b_low, inter_high, inter_low);
            c_high += inter_low;
        }
        C[row * p + col] = c_high;
    }
}

std::vector<std::vector<uint64_t>> lwr_128_64(
    const std::vector<std::vector<std::vector<uint64_t>>> &A,
    const std::vector<std::vector<std::vector<uint64_t>>> &B) {
    int n = A.size();
    int m = A[0].size();
    int p = B[0].size();

    std::vector<uint64_t> A_high(n * m), A_low(n * m);
    std::vector<uint64_t> B_high(m * p), B_low(m * p);
    std::vector<uint64_t> C(n * p, 0);

    // cudaEvent_t start_flatten, stop_flatten;
    // cudaEventCreate(&start_flatten);
    // cudaEventCreate(&stop_flatten);
    // cudaEventRecord(start_flatten, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_high[i * m + j] = A[i][j][1];
            A_low[i * m + j] = A[i][j][0];
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            B_high[i * p + j] = B[i][j][1];
            B_low[i * p + j] = B[i][j][0];
        }
    }

    // cudaEventRecord(stop_flatten, 0);
    // cudaEventSynchronize(stop_flatten);
    // float flatten_time;
    // cudaEventElapsedTime(&flatten_time, start_flatten, stop_flatten);
    // std::cout << "Flatten time: " << flatten_time << " ms" << std::endl;
    // cudaEventDestroy(start_flatten);
    // cudaEventDestroy(stop_flatten);

    // cudaEvent_t start_mem, stop_mem;
    // cudaEventCreate(&start_mem);
    // cudaEventCreate(&stop_mem);
    // cudaEventRecord(start_mem, 0);

    uint64_t *d_A_high, *d_A_low, *d_B_high, *d_B_low, *d_C;
    cudaMalloc(&d_A_high, n * m * sizeof(uint64_t));
    cudaMalloc(&d_A_low, n * m * sizeof(uint64_t));
    cudaMalloc(&d_B_high, m * p * sizeof(uint64_t));
    cudaMalloc(&d_B_low, m * p * sizeof(uint64_t));
    cudaMalloc(&d_C, n * p * sizeof(uint64_t));

    cudaMemcpy(d_A_high, A_high.data(), n * m * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_low, A_low.data(), n * m * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_high, B_high.data(), m * p * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_low, B_low.data(), m * p * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // cudaEventRecord(stop_mem, 0);
    // cudaEventSynchronize(stop_mem);
    // float mem_time;
    // cudaEventElapsedTime(&mem_time, start_mem, stop_mem);
    // std::cout << "Memory allocation and transfer time: " << mem_time << " ms" << std::endl;
    // cudaEventDestroy(start_mem);
    // cudaEventDestroy(stop_mem);

    // cudaEvent_t start_kernel, stop_kernel;
    // cudaEventCreate(&start_kernel);
    // cudaEventCreate(&stop_kernel);
    // cudaEventRecord(start_kernel, 0);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A_high, d_A_low, d_B_high, d_B_low, d_C, n, m, p);

    // cudaEventRecord(stop_kernel, 0);
    // cudaEventSynchronize(stop_kernel);
    // float kernel_time;
    // cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    // std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;
    // cudaEventDestroy(start_kernel);
    // cudaEventDestroy(stop_kernel);

    // cudaEvent_t start_transfer, stop_transfer;
    // cudaEventCreate(&start_transfer);
    // cudaEventCreate(&stop_transfer);
    // cudaEventRecord(start_transfer, 0);

    cudaMemcpyAsync(C.data(), d_C, n * p * sizeof(uint64_t), cudaMemcpyDeviceToHost, 0);

    // cudaEventRecord(stop_transfer, 0);
    // cudaEventSynchronize(stop_transfer);
    // float transfer_time;
    // cudaEventElapsedTime(&transfer_time, start_transfer, stop_transfer);
    // std::cout << "Result transfer time: " << transfer_time << " ms" << std::endl;
    // cudaEventDestroy(start_transfer);
    // cudaEventDestroy(stop_transfer);

    cudaFree(d_A_high);
    cudaFree(d_A_low);
    cudaFree(d_B_high);
    cudaFree(d_B_low);
    cudaFree(d_C);

    std::vector<std::vector<uint64_t>> result(n, std::vector<uint64_t>(p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            result[i][j] = C[i * p + j];
        }
    }
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(lwr, m) {
    m.def("lwr_128_64", &lwr_128_64, "Perform lwr_operation_128_64");
}
