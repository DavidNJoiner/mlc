#!/bin/bash

export CUDA_PATH="/usr/local/cuda-12.2"
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

NVCC_FLAGS="-arch=sm_61"
GCC_FLAGS="-I${CUDA_PATH}/include -mavx2 -march=native -O2 -fopenmp -lm"

# Compile the CUDA source file with nvcc
nvcc -g -c cuda_ops.cu -o cuda_ops.o ${NVCC_FLAGS}

# Compile the C source files with gcc
gcc -g -c config.c -o config.o ${GCC_FLAGS}
gcc -g -c mempool.c -o mempool.o ${GCC_FLAGS}
gcc -g -c main.c -o main.o ${GCC_FLAGS}
gcc -g -c device.c -o device.o ${GCC_FLAGS}
gcc -g -c avx_ops.c -o avx_ops.o ${GCC_FLAGS}
gcc -g -c data.c -o data.o ${GCC_FLAGS}
gcc -g -c debug.c -o debug.o ${GCC_FLAGS}
gcc -g -c ops.c -o ops.o ${GCC_FLAGS}
#gcc -c function.c -o function.o ${GCC_FLAGS}
gcc -g -c tensor.c -o tensor.o ${GCC_FLAGS}

# Link all the object files, including cuda.o
gcc config.o mempool.o cuda_ops.o main.o device.o avx_ops.o data.o debug.o ops.o tensor.o -L${CUDA_PATH}/lib64 -lcudart -o deepc

# Clean up object files
rm *.o
chmod +x deepc
./deepc
