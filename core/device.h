// device.h
#ifndef DEVICE_H
#define DEVICE_H

#include "config.h"
#include <stdlib.h>

typedef enum
{
    CPU,
    CUDA
} DeviceType;

typedef struct
{
    DeviceType type;
    int deviceID;
} Device;

Device*         device_init(DeviceType type, int deviceID);
void            device_release(Device *device);

#endif // _DEVICE_H_