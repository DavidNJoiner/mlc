#include "config.h"

void print_cuda_v() {
    if (deepc_cuda){
        int cudaVersion = CUDART_VERSION;
        printf("CUDA version: %d.%d detected.\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
    }else{printf("CUDA is not installed.\n");}
}

// getDevices will detect the available hardware and create a Device object for each of them.
void getDevices(){
    Device* gpu =  init_device(CUDA, 0);
    Device* cpu =  init_device(CPU, -1);
    print_cuda_v();
}
/*  -------------------------------------------------------*/
/*  Unix-like functions                                    */
/*  -------------------------------------------------------*/
/* 
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
} */

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