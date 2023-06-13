#include "tensor.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000


int calculateIndex(int* indices, int* strides, int dim) {
    int index = 0;
    for (int i = 0; i < dim; i++) {
        index += indices[i] * strides[i];
    }
    return index;
}

void flattenArray(float32* array, float32* flattened, int* shape, int dim){
    int* indices = (int*)malloc(dim * sizeof(int));
    int* strides = (int*)malloc(dim * sizeof(int));

    // Calculate strides
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Flatten array
    int flatIndex = 0;
    for (int i = 0; i < dim; i++) {
        indices[i] = 0;
    }

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            // Access array using indices and strides
            flattened[flatIndex] = array[calculateIndex(indices, strides, dim)];

            // Update indices
            for (int k = dim - 1; k >= 0; k--) {
                indices[k]++;
                if (indices[k] >= shape[k] && k > 0) {
                    indices[k] = 0;
                } else {
                    break;
                }
            }

            flatIndex++;
        }
    }

    free(indices);
    free(strides);
}



int main() {
    float32 arr[4][3] =  {{0, 0, 2},
                        {0, 1, 5},
                        {1, 0, 1},
                        {1, 1, 0}};
                        
    float32 arr2[4][3] = {{0, 0, 5},
                        {0, 1, 5},
                        {1, 0, 1},
                        {1, 1, 0}};

    int shape[] = {4, 3};
    int dim = 2;

    // Convert ndarray to Data object
    Data* data = convertToData((float32*)arr, shape, dim);
    Data* data2 = convertToData((float32*)arr2, shape, dim);

    //Tensor* t3 = createTensor(data);
    Tensor* t4 = createTensor(data);
    Tensor* res1 = createTensor(data2);

    //mult(res1,t4,t3);
    add(res1,t4);	
    printTensor(res1);

    // Deallocate memory
    free(data->values);
    free(data);
    //free(t3->gradient);
    //free(t3);
    free(t4->gradient);
    free(t4);
    free(res1->gradient);
    free(res1);

    return 0;
}