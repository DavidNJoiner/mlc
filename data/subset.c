#include "subset.h"

SubsetArray* createSubsetArray() {
    SubsetArray* arr = (SubsetArray*)malloc(sizeof(SubsetArray));
    arr->count = 0;
    arr->capacity = 2; // Initial capacity, can be adjusted as needed
    arr->subsets = (Subset*)malloc(arr->capacity * sizeof(Subset));
    return arr;
}

void addSubset(SubsetArray* arr, int* indices, int length) {
    if (arr->count == arr->capacity) {
        arr->capacity *= 2;
        arr->subsets = (Subset*)realloc(arr->subsets, arr->capacity * sizeof(Subset));
    }

    Subset subset;
    subset.indices = (int*)malloc(length * sizeof(int));
    subset.length = length;
    for (int i = 0; i < length; i++) {
        subset.indices[i] = indices[i];
    }

    arr->subsets[arr->count++] = subset;
}

void freeSubsetArray(SubsetArray* arr) {
    for (int i = 0; i < arr->count; i++) {
        free(arr->subsets[i].indices);
    }
    free(arr->subsets);
    free(arr);
}

int* randperm(int n, unsigned int seed) {
    int* indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
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

SubsetArray* random_split(Dataset* dataset, float* lengths, int lengths_size, unsigned int seed) {
    float epsilon = 1e-5;

    int total_size = dataset->data->size;
    int* sizes = (int*)malloc(lengths_size * sizeof(int));
    int remaining = total_size;

    for (int i = 0; i < lengths_size; i++) {
        float frac = lengths[i];
        if (frac < 0.0 || frac > 1.0) {
            printf("Error: Fraction at index %d is not between 0 and 1\n", i);
            free(sizes);
            return NULL;
        }
        sizes[i] = (int)floor(total_size * frac + 0.5);
        remaining -= sizes[i];
    }

    for (int i = 0; i < lengths_size; i++) {
        while (remaining > 0 && fabs(lengths[i] - (float)sizes[i] / total_size) > epsilon) {
            sizes[i]++;
            remaining--;
        }
    }

    int* indices = randperm(total_size, seed);
    SubsetArray* subsets = createSubsetArray();

    int start = 0;
    for (int i = 0; i < lengths_size; i++) {
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