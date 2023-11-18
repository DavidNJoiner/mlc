// config.h
#ifndef CONFIG_H
#define CONFIG_H

// std
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "device.h"
#include "define.h"
#include "../ops/cuda_binary_ops.h"

// OS Specific includes
#ifdef DEEPC_LINUX
#include <unistd.h>
void getDevices();
// uint32_t get_num_cores();
// void cpu_get_stats();

#elif DEEPC_WINDOWS
#include <windows.h>
//void cuda_version();
//void cpu_get_stats();
//uint32_t get_num_cores();
#endif
//------------------------------------

// Check for system architecture
#if defined(_M_X64) || defined(__amd64__)
#define DEEPC_CPU 64
#else
#define DEEPC_CPU 32
#endif
//------------------------------------

// Check for compiler version
#if __STDC_VERSION__ >= 201112L
#define C11_SUPPORTED 1
#else
#define C11_SUPPORTED 0
#endif
//------------------------------------

// Check for AVX512 support
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define SIMD_INSTRUCTION_SET "AVX512"
#define AVX512

// Check for AVX2 support
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_INSTRUCTION_SET "AVX2"
#define AVX2

// Check for AVX support
#elif defined(__AVX__)
#include <immintrin.h>
#define SIMD_INSTRUCTION_SET "AVX"
#define AVX

// Check for SSE4.2 support
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#define SIMD_INSTRUCTION_SET "SSE4.2"
#define SSE

// Check for SSE4.1 support
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define SIMD_INSTRUCTION_SET "SSE4.1"
#define SSE

// Check for SSSE3 support
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#define SIMD_INSTRUCTION_SET "SSSE3"
#define SSE

// Check for SSE3 support
#elif defined(__SSE3__)
#include <pmmintrin.h>
#define SIMD_INSTRUCTION_SET "SSE3"
#define SSE

// Check for SSE2 support
#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#include <emmintrin.h>
#define SIMD_INSTRUCTION_SET "SSE2"
#define SSE

// Check for SSE support
#elif defined(__SSE__) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#define SIMD_INSTRUCTION_SET "SSE"
#define SSE

// Check for AMD-specific instructions
#elif defined(__MMX__) || defined(__3dNOW__) || defined(__3dNOW_A__)
// Include AMD-specific headers for SIMD instructions here
// Example:
// #include <amd_simd_header.h>
#define SIMD_INSTRUCTION_SET "AMD SIMD"

#else
#define SIMD_INSTRUCTION_SET "NONE"
#warning "No SIMD instruction set support detected"

#endif


// Check for CUDA compatibility
#ifdef CUDA_AVAILABLE
#include "cuda.h"
#include "cuda_runtime.h"
void cuda_version();
#else
// Alternative non-CUDA code here
#endif

#endif // _CONFIG_H_