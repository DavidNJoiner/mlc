#!/bin/bash

gcc -o deepc main.c -mavx2 -march=native -O2 -fopenmp -lm 

./deepc
