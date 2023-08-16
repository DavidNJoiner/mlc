// device.h
#ifndef DEVICE_H_
#define DEVICE_H_

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

Device *init_device(DeviceType type, int deviceID);
void free_device(Device *device);

#endif // DEVICE_H_
