// device_manager.h
#ifndef DEVICE_MANAGER_H_
#define DEVICE_MANAGER_H_

#include <stdio.h>
#include "device.h"

void            init_dm();
void            set_device(Device* device);
int             choose_cuda_device(int* num_devices);
Device*         get_current_device();

#endif //DEVICE_MANAGER_H_
