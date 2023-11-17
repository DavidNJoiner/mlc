#include "device.h"

Device *device_init(DeviceType type, int deviceID)
{
    Device *device = (Device *)malloc(sizeof(Device));
    device->type = type;
    device->deviceID = deviceID;

    #ifdef CUDA_AVAILABLE
    cudaSetDevice(deviceID);
    #endif // CUDA_AVAILABLE

    return device;
}

void device_release(Device *device)
{
    free(device);
}