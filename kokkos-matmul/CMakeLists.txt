cmake_minimum_required(VERSION 3.16)
project(kokkos_matmul LANGUAGES CXX)

# Enable HIP support
set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "")

find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build with debug info and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

add_executable(matmul src/matmul.cpp)
target_link_libraries(matmul Kokkos::kokkos)