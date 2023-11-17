#include "arrset.h"

// Function to create a new Arrayset
arrset* createArrayset(arr_t* data, arr_t* labels) {
    arrset* dataset = (arrset*)malloc(sizeof(arrset));
    dataset->data = data;
    dataset->labels = labels;
    return dataset;
}

// Function to access elements in the Array object of the Arrayset
void** getElement(arrset* dataset, int index) {
    void** element = (void**)malloc(2 * sizeof(void*));

    if (element == NULL) {
        // Failed to allocate memory
        return NULL;
    }

    element[0] = arr_get_value_at_index(dataset->data, &index);
    element[1] = arr_get_value_at_index(dataset->labels, &index);

    return element;
}

// Function to release memory for the Arrayset
void freeArrayset(arrset* dataset) {
    free(dataset);
}
