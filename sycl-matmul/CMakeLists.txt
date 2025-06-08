cmake_minimum_required(VERSION 3.20)
project(sycl-matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find SYCL compiler
set(CMAKE_CXX_COMPILER acpp)

# Add executable
add_executable(matmul src/matmul.cpp)

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# AdaptiveCPP compiler flags
target_compile_options(matmul PRIVATE
    --acpp-targets=generic
    -fPIC
)

target_link_options(matmul PRIVATE
    --acpp-targets=generic
)

# Link OpenMP
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)

# Install rules
install(TARGETS matmul
        RUNTIME DESTINATION bin)