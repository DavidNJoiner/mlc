#include "subarrset.h"

subarrset* createSubsetArray() {
    subarrset* arr_t = (subarrset*)malloc(sizeof(subarrset));
    arr_t->count = 0;
    arr_t->capacity = 2; // Initial capacity, can be adjusted as needed
    arr_t->subsets = (subset*)malloc(arr_t->capacity * sizeof(arrset));
    return arr_t;
}

void addSubset(subarrset* arr_t, int* indices, int length) {
    if (arr_t->count == arr_t->capacity) {
        arr_t->capacity *= 2;
        arr_t->subsets = (subset*)realloc(arr_t->subsets, arr_t->capacity * sizeof(subset));
    }

    subset subset;
    subset.indices = (int*)malloc(length * sizeof(int));
    subset.length = length;
    for (uint32_t i = 0; i < length; i++) {
        subset.indices[i] = indices[i];
    }

    arr_t->subsets[arr_t->count++] = subset;
}

void freeSubsetArray(subarrset* arr_t) {
    for (uint32_t i = 0; i < arr_t->count; i++) {
        free(arr_t->subsets[i].indices);
    }
    free(arr_t->subsets);
    free(arr_t);
}

int* randperm(int n, unsigned int seed) {
    int* indices = (int*)malloc(n * sizeof(int));
    for (uint32_t i = 0; i < n; i++) {
        indices[i] = i;
    }

    srand(seed);
    for (int i = n - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    return indices;
}

subarrset* random_split(arrset* dataset, float* lengths, int lengths_size, unsigned int seed) {
    float epsilon = 1e-5;

    int total_size = dataset->data->size;
    int* sizes = (int*)malloc(lengths_size * sizeof(int));
    int remaining = total_size;

    for (uint32_t i = 0; i < lengths_size; i++) {
        float frac = lengths[i];
        if (frac < 0.0 || frac > 1.0) {
            printf("Error: Fraction at index %d is not between 0 and 1\n", i);
            free(sizes);
            return NULL;
        }
        sizes[i] = (int)floor(total_size * frac + 0.5);
        remaining -= sizes[i];
    }

    for (uint32_t i = 0; i < lengths_size; i++) {
        while (remaining > 0 && fabs(lengths[i] - (float)sizes[i] / total_size) > epsilon) {
            sizes[i]++;
            remaining--;
        }
    }

    int* indices = randperm(total_size, seed);
    subarrset* subsets = createSubsetArray();

    int start = 0;
    for (uint32_t i = 0; i < lengths_size; i++) {
        int length = sizes[i];
        int* subset_indices = (int*)malloc(length * sizeof(int));
        for (int j = 0; j < length; j++) {
            subset_indices[j] = indices[start + j];
        }
        addSubset(subsets, subset_indices, length);
        start += length;
    }

    free(indices);
    free(sizes);

    return subsets;
}