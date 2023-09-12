#!/bin/bash

PROJECT_ROOT=$(dirname "$0")
COMPILER="-std=c11"
export CUDA_PATH="/usr/local/cuda-12.2"
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export GL_LIBS="/usr/lib/x86_64-linux-gnu"

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
    GCC_FLAGS="-fPIC -I${CUDA_PATH}/include ${SIMD_FLAGS} -march=native -O2 -fopenmp -lm ${CUDA_FLAG}"
elif [ "$ARCH" == "armv7l" ]; then
    GCC_FLAGS="-fPIC -I${CUDA_PATH}/include -march=armv7-a -O2 -fopenmp -lm ${CUDA_FLAG}"
elif [ "$ARCH" == "aarch64" ]; then
    GCC_FLAGS="-fPIC -I${CUDA_PATH}/include -march=armv8-a -O2 -fopenmp -lm ${CUDA_FLAG}"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Compile the CUDA source file with nvcc
if [ -n "${NVCC_FLAGS}" ]; then
    nvcc -g -c ${PROJECT_ROOT}/ops/cuda_ops.cu -o build/cuda_ops.o ${NVCC_FLAGS}
    if [ $? -ne 0 ]; then
        echo "Compilation of cuda_ops.cu failed."
        exit 1
    fi
fi

# Compile the C source files with gcc
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/config.c -o build/config.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/deep_time.c -o build/deep_time.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/logging/table_cmd.c -o build/table_cmd.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/types/float16.c -o build/float16.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/types/dtype.c -o build/dtype.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/mempool/state_manager.c -o build/state_manager.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/mempool/pool.c -o build/pool.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/mempool/memblock.c -o build/memblock.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/mempool/subblock.c -o build/subblock.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/core/device.c -o build/device.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/ops/avx.c -o build/avx.o ${GCC_FLAGS}
#gcc ${COMPILER} -g -c ${PROJECT_ROOT}/ops/sse.c -o build/sse.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c ${PROJECT_ROOT}/ops/ops.c -o build/ops.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c data/data.c -o build/data.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c data/dataset.c -o build/dataset.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c debug.c -o build/debug.o ${GCC_FLAGS}
#gcc -c function.c -o build/function.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c tensor.c -o build/tensor.o ${GCC_FLAGS}
gcc ${COMPILER} -g -c main.c -o build/main.o ${GCC_FLAGS}

# Link all the object files, including cuda.o
if [ -n "${NVCC_FLAGS}" ]; then
    gcc build/*.o -L${CUDA_PATH}/lib64 -lcudart -o deepc -lm #-L${GL_LIBS} -lGL -lGLEW -lglut -lGLU 
else
    gcc build/*.o -L${GL_LIBS} -o deepc -lm #-lGL -lGLEW -lglut -lGLU
fi

# Clean up object files
rm build/*.o
chmod +x deepc
./deepc