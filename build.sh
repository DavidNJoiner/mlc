#!/bin/bash

COMPILER="-std=c11"
export CUDA_PATH="/usr/local/cuda-12.2"
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if [ ! -d "${CUDA_PATH}" ]; then
    echo "CUDA_PATH directory ${CUDA_PATH} does not exist. Please check the path and try again."
    exit 1
fi

# Detect if NVCC is present
if command -v nvcc &> /dev/null
then
    NVCC_FLAGS="-arch=sm_61 -DCUDA_AVAILABLE"
    CUDA_FLAG="-DCUDA_AVAILABLE"
else
    NVCC_FLAGS=""
    CUDA_FLAG=""
fi

# Detect the CPU architecture
ARCH=$(uname -m)

# Set the appropriate compiler flags based on the detected CPU architecture
if [ "$ARCH" == "x86_64" ]; then
    CPUINFO=$(grep -o -m 1 -w 'avx512\|avx2\|avx\|sse4_2\|sse4_1\|ssse3\|sse3\|sse2\|sse' /proc/cpuinfo | head -1)
    case "$CPUINFO" in
        "avx512")
            SIMD_FLAGS="-mavx512f"
            ;;
        "avx2")
            SIMD_FLAGS="-mavx2"
            ;;
        "avx")
            SIMD_FLAGS="-mavx"
            ;;
        "sse4_2")
            SIMD_FLAGS="-msse4.2"
            ;;
        "sse4_1")
            SIMD_FLAGS="-msse4.1"
            ;;
        "ssse3")
            SIMD_FLAGS="-mssse3"
            ;;
        "sse3")
            SIMD_FLAGS="-msse3"
            ;;
        "sse2")
            SIMD_FLAGS="-msse2"
            ;;
        "sse")
            SIMD_FLAGS="-msse"
            ;;
        *)
            SIMD_FLAGS=""
            ;;
    esac
    GCC_FLAGS="-I${CUDA_PATH}/include ${SIMD_FLAGS} -march=native -O2 -fopenmp -lm ${CUDA_FLAG}"
elif [ "$ARCH" == "armv7l" ]; then
    GCC_FLAGS="-I${CUDA_PATH}/include -march=armv7-a -O2 -fopenmp -lm ${CUDA_FLAG}"
elif [ "$ARCH" == "aarch64" ]; then
    GCC_FLAGS="-I${CUDA_PATH}/include -march=armv8-a -O2 -fopenmp -lm ${CUDA_FLAG}"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Compile the CUDA source file with nvcc
if [ -n "${NVCC_FLAGS}" ]; then
    nvcc -g -c cuda_ops.cu -o cuda_ops.o ${NVCC_FLAGS}
    if [ $? -ne 0 ]; then
        echo "Compilation of cuda_ops.cu failed."
        exit 1
    fi
fi

# Compile the C source files with gcc
gcc ${COMPILER} -g -c config.c -o config.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c mempool.c -o mempool.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c main.c -o main.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c device.c -o device.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c avx.c -o avx.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c dtype.c -o dtype.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c data/data.c -o data.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c data/dataset.c -o dataset.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c debug.c -o debug.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ops.c -o ops.o ${GCC_FLAGS}
#gcc -c function.c -o function.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c tensor.c -o tensor.o ${GCC_FLAGS}

# Link all the object files, including cuda.o
if [ -n "${NVCC_FLAGS}" ]; then
    gcc config.o mempool.o cuda_ops.o main.o device.o avx.o dtype.o data.o dataset.o debug.o ops.o tensor.o -L${CUDA_PATH}/lib64 -lcudart -o deepc -lm
else
    gcc config.o mempool.o main.o device.o avx.o dtype.o data.o dataset.o debug.o ops.o tensor.o -o deepc -lm
fi



# Clean up object files
rm *.o
chmod +x deepc
./deepc