#include "float16.h"

float16 float16_new()
{
    float16 h;
    h._h = 0;
    return h;
}

float16 float16_from_float(float f)
{
    float16 h;
    union
    {
        uint32_t i;
        float f;
    } uif;
    uif.f = f;

    if (f == 0.0f)
    {
        h._h = (uif.i >> 16);
    }
    else
    {
        int e = (uif.i >> 23) & 0x000001ff;
        e = _eLut[e];
        if (e)
        {
            int m = uif.i & 0x007fffff;
            h._h = e + ((m + 0x00000fff + ((m >> 13) & 1)) >> 13);
        }
        else
        {
            h._h = uif.i & 0xFFFF;
        }
    }
    return h;
}

float float16_to_float(float16 h)
{
    return _toFloat[h._h].f;
}

float16 float16_round(float16 h, unsigned int n)
{
    if (n >= 10)
    {
        return h;
    }

    uint16_t s = h._h & 0x8000;
    uint16_t e = h._h & 0x7fff;

    e >>= 9 - n;
    e += e & 1;
    e <<= 9 - n;

    if (e >= 0x7c00)
    {
        e = h._h;
        e >>= 10 - n;
        e <<= 10 - n;
    }

    float16 result;
    result._h = s | e;
    return result;
}

float16 float16_negate(float16 h)
{
    float16 result;
    result._h = h._h ^ 0x8000;
    return result;
}

void float16_assign(float16 *h, float16 other)
{
    if (h)
    {
        h->_h = other._h;
    }
}

void float16_assign_float(float16 *h, float f)
{
    if (h)
    {
        *h = float16_from_float(f);
    }
}

void float16_add_assign(float16 *h, float16 other)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        float f2 = float16_to_float(other);
        *h = float16_from_float(f1 + f2);
    }
}

void float16_add_assign_float(float16 *h, float f)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        *h = float16_from_float(f1 + f);
    }
}

void float16_sub_assign(float16 *h, float16 other)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        float f2 = float16_to_float(other);
        *h = float16_from_float(f1 - f2);
    }
}

void float16_sub_assign_float(float16 *h, float f)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        *h = float16_from_float(f1 - f);
    }
}

void float16_mul_assign(float16 *h, float16 other)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        float f2 = float16_to_float(other);
        *h = float16_from_float(f1 * f2);
    }
}

void float16_mul_assign_float(float16 *h, float f)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        *h = float16_from_float(f1 * f);
    }
}

void float16_div_assign(float16 *h, float16 other)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        float f2 = float16_to_float(other);
        *h = float16_from_float(f1 / f2);
    }
}

void float16_div_assign_float(float16 *h, float f)
{
    if (h)
    {
        float f1 = float16_to_float(*h);
        *h = float16_from_float(f1 / f);
    }
}

bool float16_isFinite(float16 h)
{
    uint16_t e = h._h & 0x7c00;
    return e != 0x7c00;
}

bool float16_isNormalized(float16 h)
{
    uint16_t e = h._h & 0x7c00;
    return e != 0 && e != 0x7c00;
}

bool float16_isDenormalized(float16 h)
{
    uint16_t e = h._h & 0x7c00;
    uint16_t m = h._h & 0x03ff;
    return e == 0 && m != 0;
}

bool float16_isZero(float16 h)
{
    uint16_t m = h._h & 0x7fff;
    return m == 0;
}

bool float16_isNan(float16 h)
{
    uint16_t e = h._h & 0x7c00;
    uint16_t m = h._h & 0x03ff;
    return e == 0x7c00 && m != 0;
}

bool float16_isInfinity(float16 h)
{
    uint16_t e = h._h & 0x7c00;
    uint16_t m = h._h & 0x03ff;
    return e == 0x7c00 && m == 0;
}

bool float16_isNegative(float16 h)
{
    return (h._h & 0x8000) != 0;
}

float16 float16_posInf()
{
    float16 result;
    result._h = 0x7c00;
    return result;
}

float16 float16_negInf()
{
    float16 result;
    result._h = 0xfc00;
    return result;
}

float16 float16_qNan()
{
    float16 result;
    result._h = 0x7fff; // Quiet NaN
    return result;
}

float16 float16_sNan()
{
    float16 result;
    result._h = 0x7dff; // Signaling NaN
    return result;
}

uint16_t float16_bits(float16 h)
{
    return h._h;
}

void float16_setBits(float16 *h, uint16_t bits)
{
    if (h)
    {
        h->_h = bits;
    }
}