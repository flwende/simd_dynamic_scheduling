// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(SIMD_H)
#define SIMD_H

#include <stdint.h>
#include <immintrin.h>

#if defined(__MIC__) || defined(__AVX512F__)
#define ALIGNMENT (64)
#if defined(STATIC_INTRINSICS) || defined(DYNAMIC_INTRINSICS)
#define VL (8)
#else
#define VL (8)
#endif
#elif defined(__AVX2__)
#define ALIGNMENT (32)
#if defined(STATIC_INTRINSICS) || defined(DYNAMIC_INTRINSICS)
#define VL (4)
#else
#define VL (4)
#endif
#else
#error "Only platforms AVX(2), KNCNI (MIC) and later are supported"
#endif

// user defined vector data types
typedef struct {
  double x[VL];
} vec_real8_t __attribute__((aligned(ALIGNMENT)));

typedef struct {
  int32_t x[VL];
} vec_int4_t __attribute__((aligned(ALIGNMENT)));

typedef struct {
  int32_t x[VL];
} vec_mask8_t __attribute__((aligned(ALIGNMENT)));

#if defined(__AVX512F__)

#pragma message ("......Compile for AVX512......")

#define __MXXd __m512d
#define __MXXi __m512i
#define __MMASK8 __mmask8

#define SIMD_SET1_INT32(X) \
  _mm512_set1_epi32(X)
#define SIMD_SET_INT32(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16) \
  _mm512_setr_epi32(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16)
#define SIMD_SET1_REAL64(X) \
  _mm512_set1_pd(X)
#define SIMD_SET_REAL64(X1,X2,X3,X4,X5,X6,X7,X8) \
  _mm512_setr_pd(X1,X2,X3,X4,X5,X6,X7,X8)

#define SIMD_INT2MASK(X) \
  _mm512_int2mask(X)
#define SIMD_MASK2INT(M) \
  _mm512_mask2int(M)

#define SIMD_AND_MASK(M1,M2) \
  _mm512_kand(M1,M2)
#define SIMD_OR_MASK(M1,M2) \
  _mm512_kor(M1,M2)
#define SIMD_NOT_MASK(M) \
  _mm512_knot(M)

#define SIMD_MASK_MOV_INT32(X0,M,X1) \
  _mm512_mask_mov_epi32(X0,M,X1)
#define SIMD_MASK_MOV_REAL64(X0,M,X1) \
  _mm512_mask_mov_pd(X0,M,X1)
#define SIMD_MASK_GATHER_REAL64(X0,M,IDX,ADDR) \
  _mm512_mask_i32logather_pd(X0,M,IDX,(void const *)ADDR,8)
#define SIMD_MASK_SCATTER_REAL64(ADDR,M,IDX,X) \
  _mm512_mask_i32loscatter_pd((void *)ADDR,M,IDX,X,8)
#define SIMD_MASK_LOAD_REAL64(X0,M,ADDR) \
  _mm512_mask_load_pd(X0,M,(void const *)ADDR)
#define SIMD_MASK_STORE_REAL64(ADDR,M,X) \
  _mm512_mask_store_pd((void *)ADDR,M,X)

#define SIMD_ADD_INT32(X1,X2) \
  _mm512_add_epi32(X1,X2)
#define SIMD_MASK_ADD_INT32(X0,M,X1,X2) \
  _mm512_mask_add_epi32(X0,M,X1,X2)
#define SIMD_MUL_REAL64(X1,X2) \
  _mm512_mul_pd(X1,X2)
#define SIMD_ADD_REAL64(X1,X2) \
  _mm512_add_pd(X1,X2)

#define SIMD_CVT_REAL64_TO_INT32(X) \
  _mm512_castsi256_si512(_mm512_cvt_roundpd_epi32(X,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC))
#define SIMD_MASK_CVT_REAL64_TO_INT32(X0,M,X1) \
  _mm512_castsi256_si512(_mm512_mask_cvt_roundpd_epi32(_mm512_castsi512_si256(X0),M,X1,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC))

#define SIMD_CMPGE_INT32(X1,X2) \
  _mm512_cmp_epi32_mask(X1,X2,_MM_CMPINT_GE)
#define SIMD_CMPLT_INT32(X1,X2) \
  _mm512_cmp_epi32_mask(X1,X2,_MM_CMPINT_LT)
#define SIMD_CMPGT_REAL64(X1,X2) \
  _mm512_cmp_pd_mask(X1,X2,_MM_CMPINT_GT)

#define SIMD_SQRT_REAL64(X) \
  _mm512_sqrt_pd(X)
#define SIMD_MASK_SQRT_REAL64(X0,M,X1) \
  _mm512_mask_sqrt_pd(X0,M,X1)
#define SIMD_MASK_LOG_REAL64(X0,M,X1) \
  _mm512_mask_log_pd(X0,M,X1)

#define SIMD_EXPAND_INT32(X,M) \
  _mm512_maskz_expand_epi32(M,X)

// macro definitions hiding KNCNI intrinsics
#elif defined(__MIC__)

#pragma message ("......Compile for KNCNI (MIC)......")

#define __MXXd __m512d
#define __MXXi __m512i
#define __MMASK8 __mmask8

#define SIMD_SET1_INT32(X) \
  _mm512_set1_epi32(X)
#define SIMD_SET_INT32(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16) \
  _mm512_setr_epi32(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16)
#define SIMD_SET1_REAL64(X) \
  _mm512_set1_pd(X)
#define SIMD_SET_REAL64(X1,X2,X3,X4,X5,X6,X7,X8) \
  _mm512_setr_pd(X1,X2,X3,X4,X5,X6,X7,X8)

#define SIMD_INT2MASK(X) \
  _mm512_int2mask(X)
#define SIMD_MASK2INT(M) \
  _mm512_mask2int(M)

#define SIMD_AND_MASK(M1,M2) \
  _mm512_kand(M1,M2)
#define SIMD_OR_MASK(M1,M2) \
  _mm512_kor(M1,M2)
#define SIMD_NOT_MASK(M) \
  _mm512_knot(M)

#define SIMD_MASK_MOV_INT32(X0,M,X1) \
  _mm512_mask_mov_epi32(X0,M,X1)
#define SIMD_MASK_MOV_REAL64(X0,M,X1) \
  _mm512_mask_mov_pd(X0,M,X1)
#define SIMD_MASK_GATHER_REAL64(X0,M,IDX,ADDR) \
  _mm512_mask_i32logather_pd(X0,M,IDX,(void const *)ADDR,8)
#define SIMD_MASK_SCATTER_REAL64(ADDR,M,IDX,X) \
  _mm512_mask_i32loscatter_pd((void *)ADDR,M,IDX,X,8)
#define SIMD_MASK_LOAD_REAL64(X0,M,ADDR) \
  _mm512_mask_load_pd(X0,M,(void const *)ADDR)
#define SIMD_MASK_STORE_REAL64(ADDR,M,X) \
  _mm512_mask_store_pd((void *)ADDR,M,X)

#define SIMD_ADD_INT32(X1,X2) \
  _mm512_add_epi32(X1,X2)
#define SIMD_MASK_ADD_INT32(X0,M,X1,X2) \
  _mm512_mask_add_epi32(X0,M,X1,X2)
#define SIMD_MUL_REAL64(X1,X2) \
  _mm512_mul_pd(X1,X2)
#define SIMD_ADD_REAL64(X1,X2) \
  _mm512_add_pd(X1,X2)

#define SIMD_CVT_REAL64_TO_INT32(X) \
  _mm512_cvtfxpnt_roundpd_epi32lo(X,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC)
#define SIMD_MASK_CVT_REAL64_TO_INT32(X0,M,X1) \
  _mm512_mask_cvtfxpnt_roundpd_epi32lo(X0,M,X1,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC)

#define SIMD_CMPGE_INT32(X1,X2) \
  _mm512_cmp_epi32_mask(X1,X2,_MM_CMPINT_GE)
#define SIMD_CMPLT_INT32(X1,X2) \
  _mm512_cmp_epi32_mask(X1,X2,_MM_CMPINT_LT)
#define SIMD_CMPGT_REAL64(X1,X2) \
  _mm512_cmp_pd_mask(X1,X2,_MM_CMPINT_GT)

#define SIMD_SQRT_REAL64(X) \
  _mm512_sqrt_pd(X)
#define SIMD_MASK_SQRT_REAL64(X0,M,X1) \
  _mm512_mask_sqrt_pd(X0,M,X1)
#define SIMD_MASK_LOG_REAL64(X0,M,X1) \
  _mm512_mask_log_pd(X0,M,X1)

__m512i simd_expand(const __m512i x, const __mmask8 m) {
  int *_x = (int *)&x;
  int _m = _mm512_mask2int(m);
  __m512i temp;
  int *_temp = (int *)&temp;
  int jj = 0;
  for (int ii=0; ii<VL; ii++)
    if (_m & (1<<ii))
      _temp[ii] = _x[jj++];
  return temp;
}

#define SIMD_EXPAND_INT32(X,M) simd_expand(X,M)

// macro definitions hiding AVX2 intrinsics
#elif defined(__AVX2__)

#pragma message ("......Compile for AVX2......")

#define __MXXd __m256d
#define __MXXi __m256i
#define __MMASK8 __m256d

#define SIMD_SET1_INT32(X) \
  _mm256_set1_epi32(X)
#define SIMD_SET_INT32(X1,X2,X3,X4,X5,X6,X7,X8) \
  _mm256_setr_epi32(X1,X2,X3,X4,X5,X6,X7,X8)
#define SIMD_SET1_REAL64(X) \
  _mm256_set1_pd(X)
#define SIMD_SET_REAL64(X1,X2,X3,X4) \
  _mm256_setr_pd(X1,X2,X3,X4)
#define SIMD_SET1_MASK(X) \
  _mm256_castsi256_pd(_mm256_set1_epi64x(X))

#define SIMD_MASK2INT(M) \
  _mm256_movemask_pd(M)

#define SIMD_AND_MASK(M1,M2) \
  _mm256_and_pd(M1,M2)
#define SIMD_OR_MASK(M1,M2) \
  _mm256_or_pd(M1,M2)
#define SIMD_NOT_MASK(M) \
  _mm256_xor_pd(M,_mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)))

__MMASK8 mask_to_lomask(const __MMASK8 m) {
  __m256 _a_ = _mm256_shuffle_ps(_mm256_castpd_ps(m), _mm256_castpd_ps(m), 0x88);
  __m256 _b_ = _mm256_permute2f128_ps(_a_, _a_, 0x3);
  return _mm256_and_pd(_mm256_castps_pd(_mm256_blend_ps(_a_, _b_, 0x3C)),_mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0x0,0x0)));
}

__MMASK8 lomask_to_mask(const __MMASK8 m) {
  __m256d _a_ = _mm256_castps_pd(_mm256_shuffle_ps(_mm256_castpd_ps(m), _mm256_castpd_ps(m), 0x50));
  __m256d _b_ = _mm256_castps_pd(_mm256_shuffle_ps(_mm256_castpd_ps(m), _mm256_castpd_ps(m), 0xFA));
  return _mm256_insertf128_pd(_a_, _mm256_extractf128_pd(_b_, 0x0), 0x1);
}

#define SIMD_MASK_MOV_INT32(X0,M,X1) \
  _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(X0),_mm256_castsi256_ps(X1),_mm256_castpd_ps(mask_to_lomask(M))))
#define SIMD_MASK_MOV_REAL64(X0,M,X1) \
  _mm256_blendv_pd(X0,X1,M)
#define SIMD_MASK_GATHER_REAL64(X0,M,IDX,ADDR) \
  _mm256_mask_i32gather_pd(X0,(double const *)ADDR,_mm256_extractf128_si256(IDX,0x0),M,8)

void mask_scatter_pd(double *addr, const __MMASK8 m, const __MXXi idx, const __MXXd x) {
  long int *_m = (long int *)&m;
  int *_idx = (int *)&idx;
  double *_x = (double *)&x;
  for (int ii=0; ii<VL; ii++) {
    if (_m[ii])
      addr[_idx[ii]] = _x[ii];
  }
}

#define SIMD_MASK_SCATTER_REAL64(ADDR,M,IDX,X) \
  mask_scatter_pd((double *)ADDR,M,IDX,X)
#define SIMD_MASK_LOAD_REAL64(X0,M,ADDR) \
  _mm256_blendv_pd(X0,_mm256_load_pd((double const *)ADDR),M)

void mask_store_pd(double *addr, const __MMASK8 m, const __MXXd x) {
  long int *_m = (long int *)&m;
  double *_x = (double *)&x;  
  for (int ii=0; ii<VL; ii++)
    if (_m[ii])
      addr[ii] = _x[ii];
}

#define SIMD_MASK_STORE_REAL64(ADDR,M,X) \
  mask_store_pd((double *)ADDR,M,X)

#define SIMD_ADD_INT32(X1,X2) \
  _mm256_add_epi32(X1,X2)
#define SIMD_MASK_ADD_INT32(X0,M,X1,X2) \
  _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(X0),_mm256_castsi256_ps(_mm256_add_epi32(X1,X2)),_mm256_castpd_ps(mask_to_lomask(M))))
#define SIMD_MUL_REAL64(X1,X2) \
  _mm256_mul_pd(X1,X2)
#define SIMD_ADD_REAL64(X1,X2) \
  _mm256_add_pd(X1,X2)

#define SIMD_CVT_REAL64_TO_INT32(X) \
  _mm256_castsi128_si256(_mm256_cvtpd_epi32(_mm256_sub_pd(X,_mm256_set1_pd(0.5))))
#define SIMD_MASK_CVT_REAL64_TO_INT32(X0,M,X1) \
  _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(X0),_mm256_castsi256_ps(_mm256_castsi128_si256(_mm256_cvtpd_epi32(_mm256_sub_pd(X1,_mm256_set1_pd(0.5))))),_mm256_castpd_ps(mask_to_lomask(M))))

#define SIMD_CMPGE_INT32(X1,X2) \
  lomask_to_mask(_mm256_castsi256_pd(_mm256_or_si256(_mm256_cmpeq_epi32(X1,X2),_mm256_cmpgt_epi32(X1,X2))))
#define SIMD_CMPLT_INT32(X1,X2) \
  lomask_to_mask(_mm256_castsi256_pd(_mm256_andnot_si256(_mm256_cmpeq_epi32(X1,X2),_mm256_cmpgt_epi32(X2,X1))))
#define SIMD_CMPGT_REAL64(X1,X2) \
  _mm256_cmp_pd(X1,X2,_CMP_GT_OS)

#define SIMD_SQRT_REAL64(X) \
  _mm256_sqrt_pd(X)
#define SIMD_MASK_SQRT_REAL64(X0,M,X1) \
  _mm256_blendv_pd(X0,_mm256_sqrt_pd(X1),M)
#define SIMD_MASK_LOG_REAL64(X0,M,X1) \
  _mm256_blendv_pd(X0,_mm256_log_pd(X1),M)

__m256i simd_expand(const __m256i x, const __m256d m) {
  int *_x = (int *)&x;
  long int *_m = (long int *)&m;
  __m256i temp;
  int *_temp = (int *)&temp;
  int jj = 0;
  for (int ii=0; ii<VL; ii++)
    if (_m[ii])
      _temp[ii] = _x[jj++];
  return temp;
}

#define SIMD_EXPAND_INT32(X,M) simd_expand(X,M)

#endif

#endif
