#!/bin/bash


gcc -c -o main.o main.c -mavx2 -march=native -O2 -fopenmp -lm -g
nvcc -c -o cuda.o cuda.cu

nvcc -o deepc main.o cuda.o
./deepc