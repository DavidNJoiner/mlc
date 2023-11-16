#ifndef ARRSET_H
#define ARRSET_H

#include "arr.h"

typedef struct ArraySet{
    arr_t* data;
    arr_t* labels;
} arrset;

arrset* createArrayset(arr_t* data, arr_t* labels);
void** getElement(arrset* dataset, int index);
void freeArrayset(arrset* dataset);

#endif //ARRSET_H