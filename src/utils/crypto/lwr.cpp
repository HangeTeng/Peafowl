#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <stdint.h>
// #include <iostream>

using uint128_t = __uint128_t;

uint128_t uint64_2_uint128(const std::vector<uint64_t> &vec) {
    return (static_cast<uint128_t>(vec[1]) << 64) | vec[0];
}

std::vector<std::vector<uint64_t>> lwr_128_64(
    const std::vector<std::vector<std::vector<uint64_t>>> &A,
    const std::vector<std::vector<std::vector<uint64_t>>> &B) {
    int n = A.size();
    int m = A[0].size();
    int p = B[0].size();

    // std::cout << "n: " << n << std::endl;
    // std::cout << "m: " << m << std::endl;
    // std::cout << "p: " << p << std::endl;


    std::vector<std::vector<uint64_t>> result(n, std::vector<uint64_t>(p, 0));
    uint128_t inter = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                inter += uint64_2_uint128(A[i][k]) * uint64_2_uint128(B[k][j]);
            }
            result[i][j] = (inter >> (128-64)) & 0xFFFFFFFFFFFFFFFFULL;
            inter = 0;
        }
    }
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(lwr_cpp, m) {
    m.def("lwr_128_64", &lwr_128_64, "Perform lwr_operation_128_64");
}
