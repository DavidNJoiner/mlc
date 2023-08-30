// device_manager.h
#ifndef DEVICE_MANAGER_H_
#define DEVICE_MANAGER_H_

#include <stdio.h>
#include "core/device.h"

void            InitDM();
void            SetDevice(Device* device);
int             SelectCudaDevice(int* num_devices);
Device*         GetCurrentDevice();

#endif //DEVICE_MANAGER_H_
