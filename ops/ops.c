#include "ops.h"
#include "cuda_binary_ops.h"
#include "intel_binary_ops.h"

/*  -------------------------------------------------------*/
/*  intel_mul_1D : Tensor data Fast Multiply Operation.    */
/*  -------------------------------------------------------*/
void intel_mul_1D(arr_t* dst, arr_t* A, arr_t* B, Device* device){
    int mat_size = dst->size;
    DeviceType device_type = device->type;
    switch (dst->dtype) {
        case FLOAT16: 
            if (device_type==CPU){
                mul_1D_f16(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_mul_float16(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
        case FLOAT32: 
            if (device_type==CPU){
                mul_1D_f32(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_mul_float32(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
        case FLOAT64: 
            if (device_type==CPU){
                mul_1D_f64(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_mul_float64(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
    }
}
/*  -------------------------------------------------------*/
/*  intel_add_1D : Tensor data Fast Add Operation.         */
/*  -------------------------------------------------------*/
void intel_add_1D(arr_t* dst, arr_t* A, Device* device){
    int mat_size = dst->size;
    DeviceType device_type = device->type;
    switch (dst->dtype) {
        case FLOAT16:
            if (device_type==CPU){
                add_1D_f16(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_add_float16(dst->values, A->values, mat_size);
                break;
            }
            break;
        case FLOAT32: 
            if (device_type==CPU){
                add_1D_f32(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_add_float32(dst->values, A->values, mat_size);
                break;
            }
            break;
        case FLOAT64: 
            if (device_type==CPU){
                add_1D_f64(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                //vec1_cuda_add_float64(dst->values, A->values, mat_size);
                break;
            }
            break;
    }
}