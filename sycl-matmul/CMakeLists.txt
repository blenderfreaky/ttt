cmake_minimum_required(VERSION 3.20)
project(sycl-matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find SYCL compiler
set(CMAKE_CXX_COMPILER acpp)

# Add executable
add_executable(matmul src/matmul.cpp)

# AdaptiveCPP compiler flags
target_compile_options(matmul PRIVATE
    -O3
    --acpp-targets=generic
    -fPIC
)

target_link_options(matmul PRIVATE
    --acpp-targets=generic
)

# Install rules
install(TARGETS matmul
        RUNTIME DESTINATION bin)