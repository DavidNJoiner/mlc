#include "data.h"
#include "debug.h"

DataPtrArray* global_data_ptr_array = NULL;

/*  -------------------------------------------------------*/
/*  dtypes functions                                       */
/*  -------------------------------------------------------*/
const char* GetDType(int dtype) {
    switch(dtype) {
        case FLOAT16: return "float16";
        case FLOAT32: return "float32";
        case FLOAT64: return "float64";
        default: return "Unknown dtype";
    }
}

int GetDTypeSize(int dtype) {
    switch (dtype) {
        case FLOAT16: return sizeof(float16);
        case FLOAT32: return sizeof(float32);
        case FLOAT64: return sizeof(float64);
        default: return 0;
    }
}
/*  -----------------------------------------------------------------------*/
/*  CalculateIndex : convert multi-dimensional index into a linear index;  */
/*  -----------------------------------------------------------------------*/
int CalculateIndex(int* indices, int* shape, int dim) {
    int index = 0;
    int stride = 1;
    for (int i = dim - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}
/*  ---------------------------------------------------------------------------------------------*/
/*  FlattenArray : recursively flattens a multi-dimensional array into a one-dimensional array.  */
/*  ---------------------------------------------------------------------------------------------*/
void FlattenArray(void* array, void* flattened, int* shape, int dim, int dtype, int idx) {
    if (dim == 0) {
        switch (dtype) {
            case FLOAT16: {
                float16* farray = (float16*)array;
                float16* fflattened = (float16*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            case FLOAT32: {
                float32* farray = (float32*)array;
                float32* fflattened = (float32*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            case FLOAT64: {
                float64* farray = (float64*)array;
                float64* fflattened = (float64*)flattened;
                fflattened[idx] = *farray;
                break;
            }
            default:
                printf("Failed to flatten : Unsupported dtype %s\n", GetDType(dtype));
                break;
        }
    } else {
        int stride = 1;
        for (int i = 1; i < dim; i++) {
            stride *= shape[i];
        }
        stride *= dtype;  // calculate the stride for the current dimension
        for (int i = 0; i < shape[0]; i++) {
            FlattenArray((char*)array + i * stride, flattened, shape + 1, dim - 1, dtype, idx + i * stride / dtype);
        }
    }
}
/*  --------------------------------------------------------------------------------*/
/*  MakeData : Converts a given multi-dimensional array into a Data structure. */
/*  --------------------------------------------------------------------------------*/
Data* MakeData(void* array, int* shape, int dim, int dtype) {
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    
    int byte_size =  size * GetDTypeSize(dtype);
    void* flattened = (float32 *)aligned_alloc(32, byte_size);

    FlattenArray(array, flattened, shape, dim, dtype, 0);

    Data* dat = (Data*)malloc(sizeof(Data));
    dat->values = flattened;
    dat->size = size;
    dat->dim = dim;
    dat->shape = shape;
    dat->dtype = dtype;

    AddDataPtr(dat); // Store pointer to that Data object 

    return dat;
}
/*  -----------------------------------------------------------------------------*/
/*  PrintData : Converts a given multi-dimensional array into a Data structure.  */
/*  -----------------------------------------------------------------------------*/
void PrintData(Data* dat) {
    const char* dtypeStr = GetDType(dat->dtype);
    if (dtypeStr == NULL) {
        printf("Error: GetDType returned NULL.\n");
        return;
    }

    printf("dtype : %4s \n", dtypeStr);
    printf("shape : ");
    for (int i = 0; i < dat->dim; i++) {
        printf("[%2d] ", dat->shape[i]);
    }
    printf("\n");
    printf("dimension : %d \n \n", dat->dim);

    if (dat->dim >= 0) {
        PrintOp(dat, dat->dim);
    } else {
        printf("Error: Invalid dimension for printOp.\n");
    }
}
/*  -----------------------------------------------------------------------------*/
/*  RandomData : Generate a Data object filled with random values.               */
/*  -----------------------------------------------------------------------------*/
Data* RandomData(int size, int min_range, int max_range, int* shape, int dim, int dtype) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));
    
    int dtypeSize = GetDTypeSize(dtype);
    int byte_size = size * dtypeSize;
    int alignment = 32; // Was 32 by default
    const char* dtypeName = GetDType(dtype);
    
    // Making sure byte_size is a multiple of the alignment
    if (byte_size % alignment != 0) {
        byte_size = ((byte_size / alignment) + 1) * alignment;
    }
    
    printf("[New Random Data] ------- size: %d, byte_size: %d, alignment: %d, dtype: %s\n", size, byte_size, alignment, dtypeName);
    void* random_values = aligned_alloc(alignment, byte_size);

    if(dtype == FLOAT16) {
        float16* ptr = (float16*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float16)(min_range + ((float16)rand() / (float16)RAND_MAX) * (max_range - min_range));
        }
    }
    if(dtype == FLOAT32) {
        float32* ptr = (float32*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float32)(min_range + ((float32)rand() / (float32)RAND_MAX) * (max_range - min_range));
        }
    } else if(dtype == FLOAT64) {
        float64* ptr = (float64*)random_values;
        for(int i = 0; i < size; i++) {
            ptr[i] = (float64)(min_range + ((float64)rand() / (float64)RAND_MAX) * (max_range - min_range));
        }
    }

    Data* data = (Data*)malloc(sizeof(Data));
    data->values = random_values;
    data->size = size;
    data->dim = dim;
    data->shape = shape;
    data->dtype = dtype;
    
    return data;
}
/*  -----------------------------------------------------------------------------*/
/*  AccessElement : Access an element in the Data object                         */
/*  -----------------------------------------------------------------------------*/
void* AccessElement(Data* data, int* indices) {
    int linear_index = CalculateIndex(indices, data->shape, data->dim);
    int dtype_size = GetDTypeSize(data->dtype);

    // Cast the data values to the correct type and return the address of the element
    switch (data->dtype) {
        case FLOAT16: {
            float16* values = (float16*)data->values;
            return (void*)&values[linear_index];
        }
        case FLOAT32: {
            float32* values = (float32*)data->values;
            return (void*)&values[linear_index];
        }
        case FLOAT64: {
            float64* values = (float64*)data->values;
            return (void*)&values[linear_index];
        }
        default:
            printf("Unsupported dtype %s\n", GetDType(data->dtype));
            return NULL;
    }
}

/*  -----------------------------------------------------------------------------*/
/*  SetElement : Set the value of an element in the Data object                  */
/*  -----------------------------------------------------------------------------*/
void SetElement(Data* data, int* indices, void* value) {
    void* element_ptr = AccessElement(data, indices);

    // Set the value of the element
    switch (data->dtype) {
        case FLOAT16: {
            float16* element = (float16*)element_ptr;
            *element = *(float16*)value;
            break;
        }
        case FLOAT32: {
            float32* element = (float32*)element_ptr;
            *element = *(float32*)value;
            break;
        }
        case FLOAT64: {
            float64* element = (float64*)element_ptr;
            *element = *(float64*)value;
            break;
        }
        default:
            printf("Failed to set element: Unsupported dtype %s\n", GetDType(data->dtype));
            break;
    }
}


/*  -----------------------------------------------------------------------------*/
/*  Memory Managment                                                             */
/*  -----------------------------------------------------------------------------*/
void InitializeGlobalDataPtrArray(int initial_capacity) {
    global_data_ptr_array = (DataPtrArray*)malloc(sizeof(DataPtrArray));
    global_data_ptr_array->data_ptrs = (Data**)malloc(sizeof(Data*) * initial_capacity);
    global_data_ptr_array->count = 0;
    global_data_ptr_array->capacity = initial_capacity;
}

void AddDataPtr(Data* data_ptr) {
    if (global_data_ptr_array->count == global_data_ptr_array->capacity) {
        global_data_ptr_array->capacity *= 2;
        global_data_ptr_array->data_ptrs = (Data**)realloc(global_data_ptr_array->data_ptrs, sizeof(Data*) * global_data_ptr_array->capacity);
    }
    global_data_ptr_array->data_ptrs[global_data_ptr_array->count++] = data_ptr;
}
void FreeAllDatas() {
    if (global_data_ptr_array != NULL) {
        for (int i = 0; i < global_data_ptr_array->count; i++) {
            if (global_data_ptr_array->data_ptrs[i] != NULL) {
                // Free the memory allocated for the 'values' field
                if (global_data_ptr_array->data_ptrs[i]->values != NULL) {
                    free(global_data_ptr_array->data_ptrs[i]->values);
                    global_data_ptr_array->data_ptrs[i]->values = NULL;
                }
                // Free the memory allocated for the 'shape' field
                if (global_data_ptr_array->data_ptrs[i]->shape != NULL) {
                    free(global_data_ptr_array->data_ptrs[i]->shape);
                    global_data_ptr_array->data_ptrs[i]->shape = NULL;
                }
                // Free the memory allocated for the Data object itself
                free(global_data_ptr_array->data_ptrs[i]);
                global_data_ptr_array->data_ptrs[i] = NULL;
            }
        }
        // Free the memory allocated for the array of Data pointers
        free(global_data_ptr_array->data_ptrs);
        global_data_ptr_array->data_ptrs = NULL;
        // Free the memory allocated for the DataPtrArray object itself
        free(global_data_ptr_array);
        global_data_ptr_array = NULL;
    }
}
