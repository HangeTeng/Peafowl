# test_lwr.py
import os
import sys
import time
import numpy as np

# Add build directory to system path
sys.path.append(os.path.abspath('build'))

import lwr_cpp
import lwr

def generate_matrices(n, m, p):
    A = np.random.randint(0, 10, (n, m, 2), dtype=np.uint64)
    B = np.random.randint(0, 10, (m, 1, 2), dtype=np.uint64)
    return A.tolist(), B.tolist()

def compare_results(res_cpp, res_cuda):
    return np.array_equal(res_cpp, res_cuda)

N = 1024 * 8  # Test matrix size
M = 1024 * 2  # Shared dimension size
P = 1  # Output matrix column size

A, B = generate_matrices(N, M, P)

# Initial verification with smaller matrices
A_test = [
    [
        [13478, 1],
        [1, 1]
    ],
    [
        [1, 1],
        [1, 1]
    ]
]

B_test = [
    [
        [1342957, 1],
    ],
    [
        [154398, 1],
    ]
]

print("Initial small matrix test:")
print("A_test:", A_test)
print("B_test:", B_test)

res_cpp_test = lwr_cpp.lwr_128_64(A_test, B_test)
res_cpp_test = np.array(res_cpp_test, dtype=np.uint64)
print("CPU result for small matrix test:", res_cpp_test)

res_cuda_test = lwr.lwr_128_64(A_test, B_test)
res_cuda_test = np.array(res_cuda_test, dtype=np.uint64)
print("GPU result for small matrix test:", res_cuda_test)

if compare_results(res_cpp_test, res_cuda_test):
    print("Results match for small matrix test!")
else:
    print("Results do not match for small matrix test.")

print("Starting performance tests with large matrices...")

# Test CPU version
start_time = time.time()
res_cpp = lwr_cpp.lwr_128_64(A, B)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time:.6f} seconds")

# Test GPU version
start_time = time.time()
res_cuda = lwr.lwr_128_64(A, B)
end_time = time.time()
gpu_time = end_time - start_time
print(f"GPU time: {gpu_time:.6f} seconds")

# Convert results to numpy arrays for comparison
res_cpp = np.array(res_cpp, dtype=np.uint64)
res_cuda = np.array(res_cuda, dtype=np.uint64)

if compare_results(res_cpp, res_cuda):
    print("Results match for large matrix test!")
else:
    print("Results do not match for large matrix test.")
