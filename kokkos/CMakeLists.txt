cmake_minimum_required(VERSION 3.30)

project(ttt CXX)

set(KOKKOS_ENABLE_HIP ON)

find_package(Kokkos REQUIRED)

add_executable(ttt-kokkos ttt.cpp)
target_link_libraries(ttt-kokkos Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
