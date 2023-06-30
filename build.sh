#!/bin/bash

export CUDA_PATH="/usr/local/cuda-12.2"
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

NVCC_FLAGS="-arch=sm_61"
GCC_FLAGS="-I${CUDA_PATH}/include -mavx2 -march=native -O2 -fopenmp -lm -g"

# Compile the CUDA source file
nvcc -c cuda.cu -o cuda.o ${NVCC_FLAGS}

nvcc -c main.c -o main.o ${NVCC_FLAGS}
nvcc -c device.c -o device.o ${NVCC_FLAGS}
gcc -c avx.c -o avx.o ${GCC_FLAGS}
nvcc -c data.c -o data.o ${NVCC_FLAGS}
nvcc -c debug.c -o debug.o ${NVCC_FLAGS}
nvcc -c ops.c -o ops.o ${NVCC_FLAGS}
nvcc -c tensor.c -o tensor.o ${NVCC_FLAGS}

# Link all the object files, including cuda_cu.o
nvcc main.o device.o cuda.o avx.o data.o debug.o ops.o tensor.o -L${CUDA_PATH}/lib64 -v -lcudart -o deepc

# Clean up object files
rm *.o
chmod +x deepc
./deepc
