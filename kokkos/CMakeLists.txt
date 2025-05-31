cmake_minimum_required(VERSION 3.31)
project(ttt
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find the Kokkos package provided by Nix
find_package(Kokkos REQUIRED)

# Add the executable
add_executable(ttt-kokkos ttt.cpp)
target_link_libraries(ttt-kokkos Kokkos::kokkos)

# Set C++ standard
set_target_properties(ttt-kokkos PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# Enable testing
enable_testing()
add_test(NAME KokkosTest COMMAND ttt-kokkos 10)
