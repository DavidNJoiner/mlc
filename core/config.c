#include "config.h"

void cuda_version()
{
    if (DEEPC_CUDA_MEMORY_CACHING)
    {
        int cudaVersion = CUDART_VERSION;
        printf("[Info] CUDA version: %d.%d.\n\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
    }
    else
    {
        printf("[Info] CUDA is not installed.\n\n");
    }
}

// getDevices will detect the available hardware and create a Device object for each of them.
void getDevices()
{
    Device *gpu = init_device(CUDA, 0);
    Device *cpu = init_device(CPU, -1);
    cuda_version();
}

/*  -------------------------------------------------------*/
/*  Unix-like functions                                    */
/*  -------------------------------------------------------*/

int get_num_core() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

void cpu_get_stats()
{
    FILE *fp;
    char buffer[128];
    char *filename = "/proc/cpuinfo";

    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    uint32_t num_cores = get_num_core();
    printf("Number of CPU cores: %d\n", num_cores);

    while (fgets(buffer, sizeof(buffer), fp) != NULL)
    {
        if (strncmp(buffer, "model name", 10) == 0)
        {
            printf("CPU Model: %s", strchr(buffer, ':') + 2);
        }
        else if (strncmp(buffer, "cpu MHz", 7) == 0)
        {
            printf("CPU Frequency: %s", strchr(buffer, ':') + 2);
        }
        else if (strncmp(buffer, "cache size", 10) == 0)
        {
            printf("Cache Size: %s", strchr(buffer, ':') + 2);
        }
    }

    fclose(fp);
}

/*  -------------------------------------------------------*/
/*  Windows functions                                      */
/*  -------------------------------------------------------*/

/* uint32_t get_num_cores()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (uint32_t)sysinfo.dwNumberOfProcessors;
}

void cpu_get_stats()
{
    // This example only gets the processor architecture
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    printf("Processor architecture: ");

    switch (sysinfo.wProcessorArchitecture)
    {
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