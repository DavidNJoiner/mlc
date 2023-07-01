#include "ops.h"


/*  -------------------------------------------------------*/
/*  speed_mul_op : Tensor Fast Multiply Operation.           */
/*  -------------------------------------------------------*/
void speed_mul_op(Data* dst, Data* A, Data* B){
    int mat_size = dst->size;
    bool use_cuda = DEEPC_CUDA;
    switch (dst->dtype) {
        case FLOAT16: 
            if (!use_cuda){
                vec1_avx_mul_float16(dst->values, A->values, B->values, mat_size);
            }else{
                vec1_cuda_mul_float16(dst->values, A->values, B->values, mat_size);
            }
            break;
        case FLOAT32: 
            if (!use_cuda){
                vec1_avx_mul_float32(dst->values, A->values, B->values, mat_size);
            }else{
                vec1_cuda_mul_float32(dst->values, A->values, B->values, mat_size);
            }
            break;
        case FLOAT64: 
            if (!use_cuda){
                vec1_avx_mul_float64(dst->values, A->values, B->values, mat_size);
            }else{
                vec1_cuda_mul_float64(dst->values, A->values, B->values, mat_size);
            }
            break;
    }
}
/*  -------------------------------------------------------*/
/*  speed_add_op : Tensor Fast Add Operation.                */
/*  -------------------------------------------------------*/
void speed_add_op(Data* dst, Data* A){
    int mat_size = dst->size;
    bool use_cuda = DEEPC_CUDA;
    switch (dst->dtype) {
        case FLOAT16: 
            if (!use_cuda){
                vec1_avx_add_float16(dst->values, A->values, mat_size);
                break;
            }else{
                vec1_cuda_add_float16(dst->values, A->values, mat_size);
                break;
            }
        case FLOAT32: 
            if (!use_cuda){
                vec1_avx_add_float32(dst->values, A->values, mat_size);
                break;
            }else{
                vec1_cuda_add_float32(dst->values, A->values, mat_size);
                break;
            }
        case FLOAT64: 
            if (!use_cuda){
                vec1_avx_add_float64(dst->values, A->values, mat_size);
                break;
            }else{
                vec1_cuda_add_float64(dst->values, A->values, mat_size);
                break;
            }
    }
}