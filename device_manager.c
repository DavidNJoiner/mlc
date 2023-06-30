#include "device_manager.h"

Device* current_device = NULL; // Global variable to store the current device

void set_device(Device* device) {
    // Store the device in the global variable
    current_device = device;
}

void init_dm() {

    /*  -------------------------------------------------------*/
    /*  Init CUDA                                              */
    /*  -------------------------------------------------------*/
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);

    if (err != cudaSuccess) {
        printf("Failed to detect CUDA devices: %s", cudaGetErrorString(err));
        set_device(init_device(CPU, -1));
    } else {
        printf("Found %d CUDA capable device(s)", num_devices);
        int best_device = choose_cuda_device(&num_devices);
        cudaSetDevice(best_device);
        set_device(init_device(CUDA, best_device));
    }
}

/* Device* get_current_device() {
    // Return the current device
    return current_device;
}

int choose_cuda_device(int* num_devices) {
    cudaDeviceProp best_prop;
    int best_device = 0;

    for (int i = 0; i < *num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Simple criterion: choose the device with the most global memory
        if (i == 0 || prop.totalGlobalMem > best_prop.totalGlobalMem) {
            best_prop = prop;
            best_device = i;
        }
    }
    return best_device;
}
 */