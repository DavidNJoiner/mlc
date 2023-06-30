#ifndef DATA_H_ 
#define DATA_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include "dtype.h"

// Data Structure
typedef struct {
    void* values;
    int size;
    int dim;
    int* shape;
    int dtype;
} Data;

// Data functions Prototypes
void            flattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx);
void            printData(Data* dat);
int             GetDtypeSize(int dtype);
int             calculateIndex(int* indices, int* strides, int dim);
const char*     GetDType(int num);
Data*           makeData(void* array, int* shape, int dim, int dtype);
Data*           randomData(int size, int* range, int* shape, int dim, int dtype);

#endif // DATA_H