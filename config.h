#ifndef CONFIG_H_
#define CONFIG_H_

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>  
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "device.h"
#include "cuda_ops.h"

/*  -------------------------------------------------------*/ 
/*  OS check / Specific Prototypes                         */
/*  -------------------------------------------------------*/
#ifdef __unix__

#include <unistd.h>

void print_cuda_v();
//uint32_t get_num_cores();
//void get_cpu_info();

/*-----------------------------------------------------*/
#elif defined(_WIN32) || defined(_WIN64)

#include <windows.h>

uint32_t get_num_cores();
void get_cpu_info();

/*-----------------------------------------------------*/
#else

#error "OS not supported!"

#endif // OS check
#endif //CONFIG_H_ 