
# import matrix_multiply

# # Example matrices
# A = [[uint128_t(1), uint128_t(2)], [uint128_t(3), uint128_t(4)]]
# B = [[uint128_t(5), uint128_t(6)], [uint128_t(7), uint128_t(8)]]

# # Call the C++ function from Python
# result = matrix_multiply.matrixMultiply(A, B)

# # Print the result
# for row in result:
#     print(row)


import matrix_multiply

# import my_module

def lwr_operation_128_64(A, B):
    return matrix_multiply.lwr_operation_128_64(A, B)


# 定义测试数据
A = [
    [
        [1<<63 -2, 1<<63 -2],
        [1<<63 -3,1<<63 - 4]
    ],
    [
        [1<<63 -5,1<<63 - 6],
        [1<<63 -7,1<<63 - 8]
    ]
]

B = [
    [
        [1<<63 -8,1<<63 - 7],
        [1<<63 -6, 1<<63 -5]
    ],
    [
        [1<<63 -4, 1<<63 -3],
        [1<<63 -2, 1<<63 -1]
    ]
]

# 调用 lwr_operation_128_64 函数
result = lwr_operation_128_64(A, B)

# 打印结果
for row in result:
    print(row)


