cmake_minimum_required(VERSION 3.11)

project(MatrixMultiplication LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

if (NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH 75)
    message("WARNING CUDA_ARCH IS BEING SET TO DEFAULT ${CUDA_ARCH}")
endif (NOT DEFINED CUDA_ARCH)

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --use_fast_math")

enable_testing()

add_executable(cpu_matrix cpu/matrix.cc)
