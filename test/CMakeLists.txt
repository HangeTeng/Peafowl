cmake_minimum_required(VERSION 3.12)
project(MatrixMultiply)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

add_library(matrix_multiply MODULE matrix_multiply.cpp)
target_link_libraries(matrix_multiply PRIVATE pybind11::module)
