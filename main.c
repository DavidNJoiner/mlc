#include "tensor.h"

#define LEARNING_RATE 0.1
#define EPOCHS 1000

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

    // Convert ndarray to Data object : convertToData currently only handle float32(float).
    Data* data = convertToData((float32*)arr, shape, dim);
    Data* data2 = convertToData((float32*)arr2, shape, dim);

    printf("dtype : %s \n", GetDType(data->dtype));
    printf("----------------------------------\n");
    
    //Tensor* t3 = createTensor(data);
    Tensor* t4 = createTensor(data);
    Tensor* res1 = createTensor(data2);
    Tensor* res2 = createTensor(data2);

    //mult(res1,t4,t3);
    add(res1,t4);
    mult(res2, res1, t4);	
    printTensor(res1);
    printf("----------------------------------\n");
    printTensor(res2);

    // Deallocate memory
    free(data->values);
    free(data);
    free(data2->values);
    free(data2);
    //free(t3->gradient);
    //free(t3);
    free(t4->gradient);
    free(t4);
    free(res1->gradient);
    free(res1);
    free(res2->gradient);
    free(res2);

    return 0;
}