#include "dtype.h"

const char* get_data_type(int dtype) {
    switch(dtype) {
        case FLOAT16: return "float16";
        case FLOAT32: return "float32";
        case FLOAT64: return "float64";
        default: return "Unknown dtype";
    }
}

int get_data_size(int dtype) {
    switch (dtype) {
        case FLOAT16: return sizeof(float16);
        case FLOAT32: return sizeof(float32);
        case FLOAT64: return sizeof(float64);
        default: return 0;
    }
}