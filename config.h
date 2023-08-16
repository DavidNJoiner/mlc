#ifndef CONFIG_H_
#define CONFIG_H_

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#include "device.h"
#include "define.h"
#include "cuda_ops.h"

// Check OS version
#ifdef DEEPC_LINUX
#include <unistd.h>
void getDevices();
// uint32_t get_num_cores();
// void get_cpu_info();

#elif DEEPC_WINDOWS
#include <windows.h>
void cuda_version();
uint32_t get_num_cores();
void get_cpu_info();
#endif

// Check for compiler version
#if __STDC_VERSION__ >= 201112L
#define C11_SUPPORTED 1
#else
#define C11_SUPPORTED 0
#endif

// Check for AVX512 support
#if defined(__AVX512F__)
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

#else
#define SIMD_INSTRUCTION_SET "NONE"
#warning "No SIMD instruction set support detected"

#endif

#ifdef CUDA_AVAILABLE
#include "cuda_runtime.h"
void cuda_version();
#else
// Alternative non-CUDA code here
#endif

#endif // CONFIG_H_