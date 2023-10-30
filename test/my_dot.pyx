# matrix_multiply.pyx
cimport cython

# 导入C++头文件
cdef extern from "matrix_multiply.cpp":
    cdef cppclass uint128_t:
        uint128_t() except +
        uint128_t(unsigned long long) except +
        uint128_t operator*(uint128_t) except +
    cdef cppclass VectorUInt128:
        VectorUInt128() except +
        void push_back(uint128_t) except +
        size_t size() except +
        uint128_t operator[](size_t) except +
    VectorUInt128 matrixMultiply(VectorUInt128, VectorUInt128)

# 将C++函数包装为Python函数
@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_matrices(list[list[int]] A, list[list[int]] B):
    cdef VectorUInt128 c_A = VectorUInt128()
    cdef VectorUInt128 c_B = VectorUInt128()
    cdef VectorUInt128 result
    cdef list[list[int]] py_result = []
    cdef list[int] row
    cdef int i, j

    # 将Python矩阵A和B转换为C++矩阵
    for row in A:
        for val in row:
            c_A.push_back(uint128_t(val))
    for row in B:
        for val in row:
            c_B.push_back(uint128_t(val))

    # 调用C++函数
    result = matrixMultiply(c_A, c_B)

    # 将C++结果转换为Python矩阵
    for i in range(result.size()):
        row = []
        for j in range(result[i]):
            row.append(int(result[i][j]))
        py_result.append(row)

    return py_result
