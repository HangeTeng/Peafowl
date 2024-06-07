# CMakeLists.txt
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(matrix_multiplication LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

add_library(lwr_cpp SHARED lwr.cpp)
target_link_libraries(lwr_cpp PRIVATE pybind11::module)
set_target_properties(lwr_cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_library(lwr SHARED lwr.cu)
target_link_libraries(lwr PRIVATE pybind11::module)
set_target_properties(lwr PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

set_property(TARGET lwr PROPERTY CUDA_ARCHITECTURES 52)  # 修改为你的GPU架构
