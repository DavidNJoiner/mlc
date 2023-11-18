cmake_minimum_required(VERSION 3.10)

project(DeepC)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA setup
find_package(CUDA 12.2 QUIET)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
    add_definitions(-DCUDA_AVAILABLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)
endif()

# CPU architecture flags
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    include(CheckCXXCompilerFlag)
    set(SIMD_FLAGS_LIST "-msse;-msse2;-msse3;-mssse3;-msse4.1;-msse4.2;-mavx;-mavx2;-mavx512f")
    foreach(flag ${SIMD_FLAGS_LIST})
        string(REPLACE "-m" "" flag_trimmed ${flag})
        string(TOUPPER ${flag_trimmed} flag_upper)
        check_cxx_compiler_flag(${flag} COMPILER_SUPPORTS_${flag_upper})
        if(COMPILER_SUPPORTS_${flag_upper})
            set(SIMD_FLAGS "${SIMD_FLAGS} ${flag}")
        endif()
    endforeach()
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
    set(SIMD_FLAGS "-march=armv7-a")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(SIMD_FLAGS "-march=armv8-a")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SIMD_FLAGS} -march=native -O2 -fopenmp -lm")
if(CUDA_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUDA_AVAILABLE")
endif()

# Source files
file(GLOB_RECURSE SOURCES "*.c")
if(CUDA_FOUND)
    cuda_add_executable(deepc ${SOURCES} cuda_binary_ops.cu)
    target_link_libraries(deepc ${CUDA_LIBRARIES})
else()
    add_executable(deepc ${SOURCES})
endif()

target_link_libraries(deepc m)
