#include "dataset.h"

// Function to create a new Dataset
Dataset* createDataset(Data* data, Data* labels) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->data = data;
    dataset->labels = labels;
    return dataset;
}

// Function to access elements in the Data object of the Dataset
void** getElement(Dataset* dataset, int index) {
    void** element = (void**)malloc(2 * sizeof(void*));

    if (element == NULL) {
        // Failed to allocate memory
        return NULL;
    }

    element[0] = data_get_element_at_index(dataset->data, &index);
    element[1] = data_get_element_at_index(dataset->labels, &index);

    return element;
}

// Function to release memory for the Dataset
void freeDataset(Dataset* dataset) {
    free(dataset);
}
