#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

    std::vector<std::vector<uint64_t>> result(n, std::vector<uint64_t>(p, 0));
    uint128_t inter;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                inter += uint64_2_uint128(A[i][k]) * uint64_2_uint128(B[k][j]);
            }
            result[i][j] = (inter >> (128-64)) & 0xFFFFFFFFFFFFFFFFULL;
        }
    }
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(lwr, m) {
    m.def("lwr_128_64", &lwr_128_64, "Perform lwr_operation_128_64");
}
