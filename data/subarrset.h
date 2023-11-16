#ifndef SUBARRSET_H
#define SUBARRSET_H

#include "arrset.h"
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct Subset
{
    int *indices;
    int length;
} subset;

typedef struct SubsetArray
{
    Subset *subsets;
    int count;
    int capacity;
} subarrset;

subarrset *createSubsetArray();
subarrset *random_split(arrset *dataset, float *lengths, int lengths_size, unsigned int seed);
void addSubset(subarrset *arr_t, int *indices, int length);
void freeSubsetArray(subarrset *arr_t);
int *randperm(int n, unsigned int seed);

#endif // SUBARRSET_H