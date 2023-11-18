#include "device.h"

Device *device_init(DeviceType type, int deviceID)
{
    Device *device = (Device *)malloc(sizeof(Device));
    device->deviceID = deviceID;

    #if defined (CUDA_AVAILABLE) && (type != CPU)
    device->type = GPU;
    cudaSetDevice(deviceID);
    #elif CUDA_AVAILABLE && (type == CPU)
    device->type = CPU;
    #endif

    #ifdef AVX  
    device->type = CPU;
    #else

    #endif // CUDA_AVAILABLE
    
    return device;
}

void device_release(Device *device)
{
    free(device);
}