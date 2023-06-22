#ifndef CONFIG_H_
#define CONFIG_H_

#include <stdint.h>

/* 
* Ensure that the pointer size is always the same as
* or smaller than the size of the largest integer type on the system. 
*/

#if DEEPC_SIZEOF_POINTER == DEEPC_SIZEOF_INT64
    #define DEEPC_SIZEOF_POINTER DEEPC_SIZEOF_INT64
#elif DEEPC_SIZEOF_POINTER == DEEPC_SIZEOF_INT32
    #define DEEPC_SIZEOF_POINTER DEEPC_SIZEOF_INT32
#else
    #define DEEPC_SIZEOF_POINTER DEEPC_SIZEOF_INT8
#endif 


/* 
* Define the system architecture on which DeepC is compiled 
*/

#if defined(__AVX__) || defined(__AVX2__)

    #if defined(__i386__) || defined(i386) || defined(_M_IX86)
        /* CPUs that support AVX/AVX2 instructions on x86 architecture */
        #define DEEPC_CPU_X86
    #elif defined(__x86_64__) || defined(__amd64__) || defined(__x86_64) || defined(_M_AMD64)
        #define DEEPC_CPU_AMD64
    #endif

#elif defined(__ARM_NEON__)
    #if defined(__arm__) && defined(__ARMEL__)
        /* CPUs that support NEON instructions on little-endian ARM architecture */
        #define DEEPC_CPU_NEON_LE
    #elif defined(__aarch64__)
        /* CPUs that support NEON instructions on AArch64 architecture */
        #define DEEPC_CPU_NEON_AARCH64
    #endif

#elif defined(__riscv) && (__riscv_xlen == 64)
    /* CPUs that support the Vector V instructions extension (RVV) on RISC-V architecture */
    #define DEEPC_CPU_RISCV

#endif

/*
* Check if you are running C11 or above
*/

#ifdef __STDC_VERSION__
    #if __STDC_VERSION__ >= 201112L
        #define HAS_C11
    #endif
#endif


#endif //CONFIG_H_ 
