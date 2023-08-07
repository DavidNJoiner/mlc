#ifndef DATASET_H
#define DATASET_H

#include "data.h"

typedef struct {
    Data* data;
    Data* labels;
} Dataset;

Dataset* createDataset(Data* data, Data* labels);
void** getElement(Dataset* dataset, int index);
void freeDataset(Dataset* dataset);

#endif //DATASET_H