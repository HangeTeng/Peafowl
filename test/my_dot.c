#include <iostream>
#include <vector>
#include <cstdint>

// Define a type for __uint128_t
using uint128_t = __uint128_t;

// Function to convert a uint128_t to a string
std::string uint128ToString(uint128_t value) {
    std::string result;
    do {
        char digit = value % 10;
        result = char('0' + digit) + result;
        value /= 10;
    } while (value > 0);
    return result;
}

// Function to multiply two matrices
std::vector<std::vector<uint128_t>> matrixMultiply(
    const std::vector<std::vector<uint128_t>>& A,
    const std::vector<std::vector<uint128_t>>& B) {
    
    int n = A.size();
    int m = A[0].size();
    int p = B[0].size();

    std::vector<std::vector<uint128_t>> result(n, std::vector<uint128_t>(p, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

int main() {
    // Define two matrices (change the values as needed)
    std::vector<std::vector<uint128_t>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<uint128_t>> B = {{5, 6}, {7, 8}};

    // Multiply the matrices
    std::vector<std::vector<uint128_t>> result = matrixMultiply(A, B);

    // Display the result
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            std::cout << uint128ToString(result[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
