#include "ops.h"

/*  -------------------------------------------------------*/
/*  speed_mul_op : Tensor data Fast Multiply Operation.    */
/*  -------------------------------------------------------*/
void speed_mul_op(Data* dst, Data* A, Data* B, Device* device){
    int mat_size = dst->size;
    DeviceType device_type = device->type;
    switch (dst->dtype) {
        case FLOAT16: 
            if (device_type==CPU){
                vec1_avx_mul_float16(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_mul_float16(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
        case FLOAT32: 
            if (device_type==CPU){
                vec1_avx_mul_float32(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_mul_float32(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
        case FLOAT64: 
            if (device_type==CPU){
                vec1_avx_mul_float64(dst->values, A->values, B->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_mul_float64(dst->values, A->values, B->values, mat_size);
                break;
            }
            break;
    }
}
/*  -------------------------------------------------------*/
/*  speed_add_op : Tensor data Fast Add Operation.         */
/*  -------------------------------------------------------*/
void speed_add_op(Data* dst, Data* A, Device* device){
    int mat_size = dst->size;
    DeviceType device_type = device->type;
    switch (dst->dtype) {
        case FLOAT16:
            if (device_type==CPU){
                vec1_avx_add_float16(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_add_float16(dst->values, A->values, mat_size);
                break;
            }
            break;
        case FLOAT32: 
            if (device_type==CPU){
                vec1_avx_add_float32(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_add_float32(dst->values, A->values, mat_size);
                break;
            }
            break;
        case FLOAT64: 
            if (device_type==CPU){
                vec1_avx_add_float64(dst->values, A->values, mat_size);
                break;
            }
            if(device_type==CUDA){
                vec1_cuda_add_float64(dst->values, A->values, mat_size);
                break;
            }
            break;
    }
}