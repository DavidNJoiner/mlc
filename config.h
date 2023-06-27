#ifndef CONFIG_H_
#define CONFIG_H_

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Dtypes
typedef float float32;
typedef double float64;
typedef unsigned short float16;
typedef unsigned short bfloat16;
typedef struct { float real; float imag; } complex32;
typedef struct { double real; double imag; } complex64;
typedef struct { long double real; long double imag; } complex128;
typedef unsigned char uint8;
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int int64;
typedef unsigned char quint8;
typedef signed char qint8;
typedef int qint32;
typedef signed char quint4x2;

//  The stride is contained within the dtype
#define FLOAT32 sizeof(float32)
#define FLOAT64 sizeof(float64)
#define FLOAT16 sizeof(float16)
#define BFLOAT16 sizeof(bfloat16)
#define COMPLEX32 sizeof(complex32)
#define COMPLEX64 sizeof(complex64)
#define COMPLEX128 sizeof(complex128)
#define UINT8 sizeof(uint8)
#define INT8 sizeof(int8)
#define INT16 sizeof(int16)
#define INT32 sizeof(int32)
#define INT64 sizeof(int64)
#define QUINT8 sizeof(quint8)
#define QINT8 sizeof(qint8)
#define QINT32 sizeof(qint32)
#define QUINT4X2 sizeof(quint4x2)

// Common Prototypes
int             GetDtypeSize(int dtype);
const char*     GetDType(int num);

/*  -------------------------------------------------------*/ 
/*  OS check / Specific Prototypes                         */
/*  -------------------------------------------------------*/

#ifdef __unix__

#include <unistd.h>

uint32_t get_num_cores();
void get_cpu_info();

/*-----------------------------------------------------*/
#elif defined(_WIN32) || defined(_WIN64)

#include <windows.h>

uint32_t get_num_cores();
void get_cpu_info();

/*-----------------------------------------------------*/
#else

#error "OS not supported!"

#endif // OS check

/*  -------------------------------------------------------*/ 
/* Define the system architecture for compilation          */
/*  -------------------------------------------------------*/

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

/*  -------------------------------------------------------*/
/* Check if you are running C11 or above                   */
/*  -------------------------------------------------------*/

#ifdef __STDC_VERSION__
    #if __STDC_VERSION__ >= 201112L
        #define HAS_C11
    #endif
#endif


#endif //CONFIG_H_ 






#ifndef CONFIG_IMPLEMENTATION
#define CONFIG_IMPLEMENTATION

/*  -------------------------------------------------------*/
/*  dtypes functions                                       */
/*  -------------------------------------------------------*/

const char* GetDType(int dtype) {
    switch(dtype) {
        case FLOAT32: return "float32";
        case FLOAT64: return "float64";
        case FLOAT16: return "float16";
        default: return "Unknown dtype";
    }
}

int GetDtypeSize(int dtype) {
    switch (dtype) {
        case FLOAT32: return sizeof(float32);
        case FLOAT64: return sizeof(float64);
        case FLOAT16: return sizeof(float16);
        // add other cases as needed
        default: return 0;
    }
}

/*  -------------------------------------------------------*/
/*  Unix-like functions                                    */
/*  -------------------------------------------------------*/

uint32_t get_num_cores() {
    return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
}

void get_cpu_info() {
    FILE* fp;
    char buffer[128];
    char* filename = "/proc/cpuinfo";

    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strncmp(buffer, "model name", 10) == 0) {
            printf("CPU Model: %s", strchr(buffer, ':') + 2);
        } else if (strncmp(buffer, "cpu MHz", 7) == 0) {
            printf("CPU Frequency: %s", strchr(buffer, ':') + 2);
        } else if (strncmp(buffer, "cache size", 10) == 0) {
            printf("Cache Size: %s", strchr(buffer, ':') + 2);
        }
    }

    fclose(fp);
}

/*  -------------------------------------------------------*/
/*  Windows functions                                      */
/*  -------------------------------------------------------*/

/* uint32_t get_num_cores() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (uint32_t)sysinfo.dwNumberOfProcessors;
}

void get_cpu_info() {
    // This example only gets the processor architecture
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    printf("Processor architecture: ");

    switch (sysinfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_INTEL:
            printf("x86");
            break;
        case PROCESSOR_ARCHITECTURE_IA64:
            printf("Intel Itanium-based");
            break;
        case PROCESSOR_ARCHITECTURE_AMD64:
            printf("x64 (AMD or Intel)");
            break;
        case PROCESSOR_ARCHITECTURE_ARM:
            printf("ARM");
            break;
        default:
            printf("Unknown architecture");
            break;
    }

    printf("\n");
} */

#endif //CONFIG_IMPLEMENTATION
