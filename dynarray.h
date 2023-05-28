#ifndef DYNARRAY_H_ 
#define DYNARRAY_H_

typedef struct {
    float* array;    // Pointer to the dynamic array
    int size;      // Current number of elements in the array
    int capacity;  // Maximum capacity of the array
} DynamicArray;


(DynamicArray* arr, float element); 
float getFromDynamicArray(DynamicArray* arr, int index);
void freeDynamicArray(DynamicArray* arr); 
void initializeArray(DynamicArray* arr, int initialCapacity);


#endif //DYNARRAY_H_

#ifndef DYNARRAY_IMPLEMENTATION
#define DYNARRAY_IMPLEMENTATION

void initializeArray(DynamicArray* arr, int initialCapacity) {
    arr->array = (float*)malloc(initialCapacity * sizeof(float));
    arr->size = 0;
    arr->capacity = initialCapacity;
}

void(DynamicArray* arr, float element) {
    if (arr->size == arr->capacity) {
        arr->capacity += arr->capacity / 2;  // Increase capacity by 50%
        arr->array = (float*)realloc(arr->array, arr->capacity * sizeof(float));
    }

    arr->array[arr->size] = element;
    arr->size++;
}

float getFromDynamicArray(DynamicArray* arr, int index) {
    if (index < 0 || index >= arr->size) {
            // Handle index out of bounds error
            // For simplicity, let's assume the array is indexed from 0 to size-1
            // You can choose your error handling approach here
            // (e.g., print an error message and return a default value)
    }

    return arr->array[index];
}

void freeDynamicArray(DynamicArray* arr) {
    free(arr->array);
    arr->array = NULL;
    arr->size = 0;
    arr->capacity = 0;
}
#endif //DYNARRAY_IMPLEMENTATION
