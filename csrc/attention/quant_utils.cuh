#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "attention_dtypes.h"
#include "dtype_float16.cuh"
#include "dtype_float32.cuh"
using namespace vllm;

/*
struct PseudoHalf {
    uint16_t bits;
};

struct PseudoFP8 {
    uint8_t bits;
};*/

typedef uint8_t PseudoFP8;
typedef uint16_t PseudoHalf;

inline __device__ PseudoFP8 half_to_fp8e4m3(PseudoHalf fp16) {
    PseudoFP8 fp8;
    // IEEE(half): 1,e5,m10
    uint8_t sign = (fp16 >> 15) & 0x1;
    uint8_t exponent = ((fp16 >> 10) & 0x1F) - 31;  // 指数减去偏移值
    uint8_t fraction = (fp16 >> 7) & 0x07; // 保留 3 位尾数
    // 使用条件表达式处理上溢和下溢的情况
    exponent = (exponent > 7) ? 15 : ((exponent < -8) ? 0 : exponent + 7);
    fraction = (exponent > 7 || exponent < -8) ? 0 : fraction;

    fp8 = (sign << 7) | (exponent << 3) | fraction;
    return fp8;
}

inline __device__ PseudoHalf fp8e4m3_to_half(PseudoFP8 fp8) {
    PseudoHalf fp16;
    uint8_t sign = (fp8 >> 7) & 0x1;
    uint8_t exponent = ((fp8 >> 3) & 0x0F) - 15 + 31;  // 指数矫正偏移值
    uint8_t fraction = fp8 & 0x07; // 保留 3 位尾数

    // IEEE(half): 1,e5,m10
    fp16 = (sign << 15) | (exponent << 10) | (fraction << 7);
    return fp16;
}

inline __device__ PseudoFP8 half_to_fp8e3m5(PseudoHalf fp16) {
    PseudoFP8 fp8;
    // IEEE(half): 1,e5,m10
    uint8_t sign = (fp16 >> 15) & 0x1;
    uint8_t exponent = ((fp16 >> 10) & 0x1F) - 31;  // 指数减去偏移值
    uint8_t fraction = (fp16 >> 5) & 0x1F; // 保留 5 位尾数
    // 使用条件表达式处理上溢和下溢的情况
    exponent = (exponent > 3) ? 7 : ((exponent < -4) ? 0 : exponent + 3);
    fraction = (exponent > 3 || exponent < -4) ? 0 : fraction;

    fp8 = (sign << 7) | (exponent << 5) | fraction;
    return fp8;
}


inline __device__ PseudoFP8 half_to_fp8e5m2(PseudoHalf fp16) {
    PseudoFP8 fp8;
    // IEEE(half): 1,e5,m10
    fp8 = (fp16 >> 8) & 0xFF;
    return fp8;
}


template<typename Tout, typename Tin>
inline __device__ Tout fp8e5m2_to_half(Tin x)
{
    return x;
}

inline __device__ uint16_t __fp8e5m2_to_half(uint8_t fp8) {
    PseudoHalf fp16;
    // IEEE(half): 1,e5,m10
    fp16 = fp8 << 8;
    return fp16;
}

template<>
inline __device__ uint16_t fp8e5m2_to_half<uint16_t, uint8_t>(uint8_t fp8) {
    PseudoHalf fp16;
    // IEEE(half): 1,e5,m10
    fp16 = fp8 << 8;
    return fp16;
}

template<>
inline __device__ uint32_t fp8e5m2_to_half<uint32_t, uint16_t>(uint16_t d) {
    union {
        uint16_t fp16x2[2];
        uint32_t uint32;
    };
    union {
        uint8_t fp8x2[2];
        uint16_t uint16;
    };
    uint16 = d;
    fp16x2[0] = __fp8e5m2_to_half(fp8x2[0]);
    fp16x2[1] = __fp8e5m2_to_half(fp8x2[1]);
    return uint32;
}

template<>
inline __device__  uint2 fp8e5m2_to_half<uint2, uint32_t>(uint32_t d) {
    union {
        uint16_t fp16x4[4];
        uint2 uint64;
    };
    union {
        uint8_t fp8x4[4];
        uint32_t uint32;
    };
    uint32 = d;
    fp16x4[0] = __fp8e5m2_to_half(fp8x4[0]);
    fp16x4[1] = __fp8e5m2_to_half(fp8x4[1]);
    fp16x4[2] = __fp8e5m2_to_half(fp8x4[2]);
    fp16x4[3] = __fp8e5m2_to_half(fp8x4[3]);
    return uint64;
}

template<>
inline __device__ uint64_t fp8e5m2_to_half<uint64_t, uint32_t>(uint32_t d) {
    union {
        uint16_t fp16x4[4];
        uint64_t uint64;
    };
    union {
        uint8_t fp8x4[4];
        uint32_t uint32;
    };
    uint32 = d;
    fp16x4[0] = __fp8e5m2_to_half(fp8x4[0]);
    fp16x4[1] = __fp8e5m2_to_half(fp8x4[1]);
    fp16x4[2] = __fp8e5m2_to_half(fp8x4[2]);
    fp16x4[3] = __fp8e5m2_to_half(fp8x4[3]);
    return uint64;
}

template<>
inline __device__ uint4 fp8e5m2_to_half<uint4, uint64_t>(uint64_t d) {
    union {
        uint16_t fp16x8[8];
        uint4 uint64x2;
    };
    union {
        uint8_t fp8x8[8];
        uint64_t uint64;
    };
    uint64 = d;
    fp16x8[0] = __fp8e5m2_to_half(fp8x8[0]);
    fp16x8[1] = __fp8e5m2_to_half(fp8x8[1]);
    fp16x8[2] = __fp8e5m2_to_half(fp8x8[2]);
    fp16x8[3] = __fp8e5m2_to_half(fp8x8[3]);
    fp16x8[4] = __fp8e5m2_to_half(fp8x8[4]);
    fp16x8[5] = __fp8e5m2_to_half(fp8x8[5]);
    fp16x8[6] = __fp8e5m2_to_half(fp8x8[6]);
    fp16x8[7] = __fp8e5m2_to_half(fp8x8[7]);
 
    return uint64x2;
}

template <int VEC_SIZE>
union QuantParamVec {};

template <>
union QuantParamVec<1> {
    float params[1];
    float data;
};

template <>
union QuantParamVec<2> {
    float params[2];
    float2 data;
};

template <>
union QuantParamVec<4> {
    float params[4];
    Float4_ data;
};

template <>
union QuantParamVec<8> {
    float params[8];
    Float8_ data;
};

// this function is for function matching, delete it after writing customized dispatch functions
inline __device__ int8_t quant(double a, const float scale, const float zp)
{
    int8_t int8;
    int8 = round(max(-128.f, min(127.f, (a - zp) / scale)));
    return int8;
}

inline __device__ int8_t quant(float a, const float scale, const float zp)
{
    int8_t int8;
    int8 = round(max(-128.f, min(127.f, (a - zp) / scale)));
    return int8;
}

inline __device__ short quant(float2 a, const float scale, const float zp)
{
    union {
        int8_t int8[2];
        short  int16;
    };

    int8[0] = round(max(-128.f, min(127.f, (a.x - zp) / scale)));
    int8[1] = round(max(-128.f, min(127.f, (a.y - zp) / scale)));
    return int16;
}

inline __device__ int32_t quant(float4 a, const float scale, const float zp)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };

    int8[0] = round(max(-128.f, min(127.f, (a.x - zp) / scale)));
    int8[1] = round(max(-128.f, min(127.f, (a.y - zp) / scale)));
    int8[2] = round(max(-128.f, min(127.f, (a.z - zp) / scale)));
    int8[3] = round(max(-128.f, min(127.f, (a.w - zp) / scale)));
    return int32;
}

// float16 to int8
inline __device__ int8_t quant(uint16_t a, const float scale, const float zp)
{
    int8_t int8;
    float  b = half_to_float(a);
    int8     = round(max(-128.f, min(127.f, (b - zp) / scale)));
    return int8;
}

// float16x2 to int8x2
inline __device__ int16_t quant(uint32_t a, const float scale, const float zp)
{
    union {
        int8_t int8[2];
        short  int16;
    };
    float2 b = half2_to_float2(a);

    int8[0] = round(max(-128.f, min(127.f, (b.x - zp) / scale)));
    int8[1] = round(max(-128.f, min(127.f, (b.y - zp) / scale)));
    return int16;
}

// float16x4 to int8x4
inline __device__ int32_t quant(uint2 a, const float scale, const float zp)
{
    union {
        int16_t int16[2];
        int32_t int32;
    };

    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    return int32;
}

// float16x8 to int8x8
inline __device__ int64_t quant(uint4 a, const float scale, const float zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };

    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    int16[2] = quant(a.z, scale, zp);
    int16[3] = quant(a.w, scale, zp);
    return int64;
}


// int8 to float32, then `vec_conversion` to target format
inline __device__ float dequant(int8_t a, const float scale, const float zp)
{
    float b = a * scale + zp;
    return b;
}


// int8x2 to float32x2
inline __device__ float2 dequant(int16_t a, const float scale, const float zp)
{
    union {
        int8_t  int8[2];
        int16_t int16;
    };
    int16 = a;

    float2 b;
    b.x = int8[0] * scale + zp;
    b.y = int8[1] * scale + zp;
    return b;
}


// int8x2 to float32x2
inline __device__ float2 dequant(int16_t a, const float2 scale, const float2 zp)
{
    union {
        int8_t  int8[2];
        int16_t int16;
    };
    int16 = a;

    float2 b;
    b.x = int8[0] * scale.x + zp.x;
    b.y = int8[1] * scale.y + zp.y;
    return b;
}

// int8x4 to float32x4
inline __device__ Float4_ dequant(int32_t a, const float scale, const float zp)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };
    int32 = a;

    Float4_ b;
    b.x.x = (int8[0] * scale) + zp;
    b.x.y = (int8[1] * scale) + zp;
    b.y.x = (int8[2] * scale) + zp;
    b.y.y = (int8[3] * scale) + zp;
    return b;
}

// int8x4 to float32x4
inline __device__ Float4_ dequant(int32_t a, const Float4_ scale, const Float4_ zp)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };
    int32 = a;

    Float4_ b;
    b.x.x = (int8[0] * scale.x.x) + zp.x.x;
    b.x.y = (int8[1] * scale.x.y) + zp.x.y;
    b.y.x = (int8[2] * scale.y.x) + zp.y.x;
    b.y.y = (int8[3] * scale.y.y) + zp.y.y;
    return b;
}

inline __device__ Float8_ dequant(int64_t a, const float scale, const float zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };
    int64 = a;

    Float8_ b;
    b.x = dequant(int16[0], scale, zp);
    b.y = dequant(int16[1], scale, zp);
    b.z = dequant(int16[2], scale, zp);
    b.w = dequant(int16[3], scale, zp);
    return b;
}

inline __device__ Float8_ dequant(int64_t a, const Float8_ scale, const Float8_ zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };
    int64 = a;

    Float8_ b;
    b.x = dequant(int16[0], scale.x, zp.x);
    b.y = dequant(int16[1], scale.y, zp.y);
    b.z = dequant(int16[2], scale.z, zp.z);
    b.w = dequant(int16[3], scale.w, zp.w);
    return b;
}

inline __device__ float __cast_to_float(const __nv_bfloat16 val) {
    return __bfloat162float(val);
}

inline __device__ float __cast_to_float(const float val) {
    return val;
}

inline __device__ float __cast_to_float(const uint16_t val) {
    return half_to_float(val);
}



template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2    float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(a);
    return uint32;
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2  b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x   = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y   = vec_conversion<uint32_t, float2>(val);

    return b;
}

template<>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)
{
    float4 b;
    b.x = a.x.x;
    b.y = a.x.y;
    b.z = a.y.x;
    b.w = a.y.y;
    return b;
}

template<>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)
{
    uint4 b;
    b.x = vec_conversion<uint32_t, float2>(a.x);
    b.y = vec_conversion<uint32_t, float2>(a.y);
    b.z = vec_conversion<uint32_t, float2>(a.z);
    b.w = vec_conversion<uint32_t, float2>(a.w);
    return b;
}

template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(const float2 &a) {
    return __float22bfloat162_rn(a);
}

template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_ &a) {
    bf16_4_t b;
    b.x = vec_conversion<__nv_bfloat162, float2>(a.x);
    b.y = vec_conversion<__nv_bfloat162, float2>(a.y);
    return b;
}

template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_ &a) {
    bf16_8_t b;
    b.x = vec_conversion<__nv_bfloat162, float2>(a.x);
    b.y = vec_conversion<__nv_bfloat162, float2>(a.y);
    b.z = vec_conversion<__nv_bfloat162, float2>(a.z);
    b.w = vec_conversion<__nv_bfloat162, float2>(a.w);
    return b;
}

static inline __device__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

#if ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}

template<>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16
