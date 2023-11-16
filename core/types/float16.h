#ifndef _FLOAT16_H_
#define _FLOAT16_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// Define the float16 type as a struct in C
typedef struct
{
    uint16_t _h;
} float16;

typedef union
{
    unsigned int i;
    float f;
} uif;

static uif _toFloat[1 << 16];
static unsigned short _eLut[1 << 9];

//-------------------------------------------------------------------------
// Limits
//
// Visual C++ will complain if HALF_MIN, HALF_NRM_MIN etc. are not float
// constants, but at least one other compiler (gcc 2.96) produces incorrect
// results if they are.
//-------------------------------------------------------------------------
#if DEEPC_WINDOWS && defined _MSC_VER
#define HALF_MIN 5.96046448e-08f     // Smallest positive float16
#define HALF_NRM_MIN 6.10351562e-05f // Smallest positive normalized float16
#define HALF_MAX 65504.0f            // Largest positive float16
#define HALF_EPSILON 0.00097656f     // Smallest positive e for which
                                     // float16 (1.0 + e) != float16 (1.0)
#elif DEEPC_LINUX
#define HALF_MIN 5.96046448e-08     // Smallest positive float16
#define HALF_NRM_MIN 6.10351562e-05 // Smallest positive normalized float16
#define HALF_MAX 65504.0            // Largest positive float16
#define HALF_EPSILON 0.00097656     // Smallest positive e for which
// float16 (1.0 + e) != float16 (1.0)
#endif

#define HALF_MANT_DIG 11   // Number of digits in mantissa
                           // (significand + hidden leading 1)
#define HALF_DIG 2         // Number of base 10 digits that
                           // can be represented without change
#define HALF_RADIX 2       // Base of the exponent
#define HALF_MIN_EXP -13   // Minimum negative integer such that
                           // HALF_RADIX raised to the power of
                           // one less than that integer is a
                           // normalized float16
#define HALF_MAX_EXP 16    // Maximum positive integer such that
                           // HALF_RADIX raised to the power of
                           // one less than that integer is a
                           // normalized float16
#define HALF_MIN_10_EXP -4 // Minimum positive integer such
                           // that 10 raised to that power is
                           // a normalized float16
#define HALF_MAX_10_EXP 4  // Maximum positive integer such
                           // that 10 raised to that power is
                           // a normalized float16

// Function prototypes
float16 float16_add(float16 a, float16 b);
float16 float16_mult(float16 a, float16 b);
float16 float16_new(); // Constructor equivalent
float16 float16_from_float(float f);
float float16_to_float(float16 h);
float16 float16_round(float16 h, unsigned int n);
float16 float16_negate(float16 h);
void float16_assign(float16 *h, float16 other);
void float16_assign_float(float16 *h, float f);
void float16_add_assign(float16 *h, float16 other);
void float16_add_assign_float(float16 *h, float f);
void float16_sub_assign(float16 *h, float16 other);
void float16_sub_assign_float(float16 *h, float f);
void float16_mul_assign(float16 *h, float16 other);
void float16_mul_assign_float(float16 *h, float f);
void float16_div_assign(float16 *h, float16 other);
void float16_div_assign_float(float16 *h, float f);
bool float16_isFinite(float16 h);
bool float16_isNormalized(float16 h);
bool float16_isDenormalized(float16 h);
bool float16_isZero(float16 h);
bool float16_isNan(float16 h);
bool float16_isInfinity(float16 h);
bool float16_isNegative(float16 h);
float16 float16_posInf();
float16 float16_negInf();
float16 float16_qNan();
float16 float16_sNan();
uint16_t float16_bits(float16 h);
void float16_setBits(float16 *h, uint16_t bits);

#endif //_FLOAT16_H