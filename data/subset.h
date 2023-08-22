#ifndef SUBSET_H
#define SUBSET_H

#include "dataset.h"
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct
{
    int *indices;
    int length;
} Subset;

typedef struct
{
    Subset *subsets;
    int count;
    int capacity;
} SubsetArray;

SubsetArray *createSubsetArray();
SubsetArray *random_split(Dataset *dataset, float *lengths, int lengths_size, unsigned int seed);
void addSubset(SubsetArray *arr, int *indices, int length);
void freeSubsetArray(SubsetArray *arr);
int *randperm(int n, unsigned int seed);

#endif // SUBSET_H