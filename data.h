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
void            FlattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx);
void            PrintData(Data* dat);
int             GetDTypeSize(int dtype);
int             CalculateIndex(int* indices, int* strides, int dim);
const char*     GetDType(int num);
Data*           MakeData(void* array, int* shape, int dim, int dtype);
Data*           RandomData(int size, int min_range, int max_range, int* shape, int dim, int dtype);
void*           AccessElement(Data* data, int* indices);
void            SetElement(Data* data, int* indices, void* value);

#endif // DATA_H