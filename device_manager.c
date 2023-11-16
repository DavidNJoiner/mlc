#include "device_manager.h"

Device *current_device = NULL; // Global variable to store the current device

void SetDevice(Device *device)
{
    current_device = device;
}

void InitDM()
{
    /*  -------------------------------------------------------*/
    /*  Init CUDA                                              */
    /*  -------------------------------------------------------*/
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);

    if (err != cudaSuccess)
    {
        printf("Failed to detect CUDA devices: %s", cudaGetErrorString(err));
        SetDevice(init_device(CPU, -1));
    }
    else
    {
        printf("Found %d CUDA capable device(s)", num_devices);
        int best_device = SelectCudaDevice(&num_devices);
        cudaSetDevice(best_device);
        SetDevice(init_device(CUDA, best_device));
    }
}

Device *GetCurrentDevice()
{
    // Return the current device
    return current_device;
}

int SelectCudaDevice(int *num_devices)
{
    cudaDeviceProp best_prop;
    int best_device = 0;

    for (uint32_t i = 0; i < *num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Simple criterion: choose the device with the most global memory
        if (i == 0 || prop.totalGlobalMem > best_prop.totalGlobalMem)
        {
            best_prop = prop;
            best_device = i;
        }
    }
    return best_device;
}