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

#define DEEPC_NO_CUDA_MEMORY_CACHING 1
#define MAX_OBJ_PER_BLOCK 2 
#define DEEPC_SIZE_OF_VOID_POINTER sizeof(void*)

typedef enum {
    TENSOR,
    DEVICE,
    DATA,
    FUNCTION,
    NEURON,
    LAYER,
    NN,
    PARAMETERS,
    LAST_FUNCTION_SUBCLASS = PARAMETERS,
} ObjectType;

/*  -------------------------------------------------------*/ 
/*  OS check / Specific Prototypes                         */
/*  -------------------------------------------------------*/
#ifdef __unix__

#include <unistd.h>

void cuda_version();
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