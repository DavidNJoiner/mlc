#include "device.h"

Device *init_device(DeviceType type, int deviceID)
{
    Device *device = (Device *)malloc(sizeof(Device));
    device->type = type;
    device->deviceID = deviceID;

    #ifdef CUDA_AVAILABLE
    cudaSetDevice(deviceID);
    #endif // CUDA_AVAILABLE

    return device;
}

void free_device(Device *device)
{
    free(device);
}