cmake_minimum_required(VERSION 3.16)
project(kokkos_matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable HIP backend and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# HIP and Kokkos configuration
set(Kokkos_ENABLE_HIP ON)
set(Kokkos_ENABLE_SERIAL OFF)
set(Kokkos_ENABLE_OPENMP ON)
set(Kokkos_ARCH_VEGA906 ON)

# Find Kokkos package with HIP
find_package(Kokkos REQUIRED)
if(NOT Kokkos_ENABLE_HIP)
    message(FATAL_ERROR "Kokkos was not built with HIP support")
endif()

# Add executable and link with Kokkos
add_executable(matmul src/matmul.cpp)
target_link_libraries(matmul Kokkos::kokkos)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Kokkos_ENABLE_HIP: ${Kokkos_ENABLE_HIP}")
message(STATUS "Kokkos_ENABLE_OPENMP: ${Kokkos_ENABLE_OPENMP}")
message(STATUS "ROCM_PATH: $ENV{ROCM_PATH}")