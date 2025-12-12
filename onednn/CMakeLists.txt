cmake_minimum_required(VERSION 4.0)
project(ttt_onednn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(INTEL_LLVM_DIR "")
if(DEFINED ENV{INTEL_LLVM_DIR})
    set(INTEL_LLVM_DIR $ENV{INTEL_LLVM_DIR})

    set(CMAKE_C_COMPILER ${INTEL_LLVM_DIR}/bin/clang)
    set(CMAKE_CXX_COMPILER ${INTEL_LLVM_DIR}/bin/clang++)
endif()

# ROCm paths should be set via ROCM_PATH and ROCM_DEVICE_LIB_PATH environment variables
set(ROCM_PATH_ARG "")
set(ROCM_DEVLIB_ARG "")
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH_ARG "--rocm-path=$ENV{ROCM_PATH}")
else()
    message(FATAL_ERROR "ROCM_PATH environment variable is not set")
endif()
if(DEFINED ENV{ROCM_DEVICE_LIB_PATH})
    set(ROCM_DEVLIB_ARG "--rocm-device-lib-path=$ENV{ROCM_DEVICE_LIB_PATH}")
else()
    message(FATAL_ERROR "ROCM_DEVICE_LIB_PATH environment variable is not set")
endif()

if(DEFINED ENV{SPIRV_TOOLS_PATH})
    set(SPIRV_TOOLS_PATH_ARG "-fsycl-libspirv-path=$ENV{SPIRV_TOOLS_PATH}")
else()
    message(FATAL_ERROR "SPIRV_TOOLS_PATH environment variable is not set")
endif()

set(PROFILING_FLAGS "")
if(DEFINED TTT_PROFILING)
    set(PROFILING_FLAGS "-fsycl-profile -gline-tables-only -fdebug-info-for-profiling")
endif()

set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1030 ${ROCM_PATH_ARG} ${ROCM_DEVLIB_ARG} -O2 -g ${PROFILING_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-fsycl ${ROCM_PATH_ARG} ${ROCM_DEVLIB_ARG}")
set(CMAKE_SHARED_LINKER_FLAGS "-fsycl ${ROCM_PATH_ARG} ${ROCM_DEVLIB_ARG}")

find_library(DNNL_LIBRARY dnnl)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${DNNL_INCLUDE_DIR}
)

# TTT Library
add_library(ttt_lib STATIC
    src/kernels.cpp
    src/ttt.cpp
)

target_link_libraries(ttt_lib
    ${DNNL_LIBRARY}
    sycl
)

# Example/Test executable
add_executable(ttt_example
    src/main.cpp
)

target_link_libraries(ttt_example
    ttt_lib
    ${DNNL_LIBRARY}
    sycl
)

enable_testing()
add_test(NAME ttt_example_test COMMAND ttt_example)
