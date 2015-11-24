// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "simd.h"

#define N (8 * 1024 * 1024) // array length
#define D (50) // maximum number of while-loop iterations (D: depth)

//#define VECTORIZE // use OpenMP4 pragmas for vectorization (explicit vectorization)
#define GREEDY_SCHEDULE // use vector expand operation for dynamic scheduling
#define CHECK_RESULTS // compare against non-SIMD version

//////////////////////////////////////////////////////
// the kernel: 
// ===========
// input:  x1[] (uniform over [0.0,2.0])
//         x2[] (uniform over [0.0,1.0])
// output: y[]
// 
// for (int32_t i=0; i<N; i++) {
//   int32_t k = 0;
//   int32_t kmax = (int32_t)(D * x2[i]);
//   double temp_y = 0.0;
//   double temp_x1 = x1[i];
//   while (k < kmax) {
//     temp_y = sqrt(temp_x1 + temp_y);
//     if (temp_y > 1.0) temp_y = log(temp_y);
//     k++;
//   }
//   y[i] = temp_y;
// }
//////////////////////////////////////////////////////

int32_t main() {

  // allocate memory with appropriate alignment
  double *x1 = (double *) _mm_malloc(N * sizeof(double), ALIGNMENT);
  double *x2 = (double *) _mm_malloc(N * sizeof(double), ALIGNMENT);
  double *y = (double *) _mm_malloc(N * sizeof(double), ALIGNMENT);
  
  srand48(1);
  for (int32_t i=0; i<N; i++) {
    x1[i] = 2.0 * drand48(); // range [0.0,2.0]
    x2[i] = drand48(); // range [0.0,1.0]
  }

  double time = omp_get_wtime();

#if defined(STATIC_MANUALVEC)

#pragma message ("......Compile STATIC_MANUALVEC version......")

  vec_real8_t vx1, vx2, vy;
  vec_mask8_t m0;

  for (int32_t i=0; i<N; i+=VL) {
    // prologue
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd safelen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      m0.x[ii] = 0x0;
      if ((i + ii) < N) {
        m0.x[ii] = 0x1;
        vx1.x[ii] = x1[i+ii];
        vx2.x[ii] = x2[i+ii];
      }
    }
    // compute
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd safelen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      int32_t k = 0;
      int32_t kmax = (int32_t)(D * vx2.x[ii]);
      double dtemp_1 = 0.0;
      while (m0.x[ii] && k < kmax) {
        dtemp_1 = sqrt(vx1.x[ii] + dtemp_1);
	double dtemp_2 = log(dtemp_1);
        if (dtemp_1 > 1.0) 
	  dtemp_1 = dtemp_2;
        k++;
      }
      vy.x[ii] = dtemp_1;
    } 
    // epilogue
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd safelen(VL)
#endif                                                                                            
    for (int32_t ii=0; ii<VL; ii++)
      if (m0.x[ii]) 
	y[i+ii] = vy.x[ii];
  }

#elif defined(DYNAMIC_MANUALVEC) && (defined(__MIC__) || defined(__AVX512F__))

#pragma message ("......Compile DYNAMIC_MANUALVEC KNCNI / AVX512 version......")

  vec_int4_t lane_alife, lane_acquire_work;
  vec_int4_t idx, k, kmax;
  vec_real8_t vx1, vx2, vy, dtemp;
  double temp;
  int32_t lane_alife_any, acquire_work_any;
#if defined(GREEDY_SCHEDULE)
  int32_t icurrent = VL;
#endif

  // prologue
  lane_alife_any = 0x1;
  acquire_work_any = 0x1;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd safelen(VL)
#endif
  for (int32_t ii=0; ii<VL; ii++) {
    lane_alife.x[ii] = 0x1;
    lane_acquire_work.x[ii] = 0x1;
    idx.x[ii] = ii;
  }

  // compute + epilogue (acquire new work)
  while (lane_alife_any) {
    if (acquire_work_any) {
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd safelen(VL)
#endif
      for (int32_t ii=0; ii<VL; ii++) {
        if (lane_alife.x[ii] && lane_acquire_work.x[ii]) {
          vx2.x[ii] = x2[idx.x[ii]];
	  vx1.x[ii] = x1[idx.x[ii]];
          vy.x[ii] = 0.0;
          k.x[ii] = 0;
          kmax.x[ii] = (int32_t)(D * vx2.x[ii]);
	  if (k.x[ii] < kmax.x[ii]) {
	    lane_acquire_work.x[ii] = 0x0;
	  }
        }
      }
    }
    acquire_work_any = 0x0;
    lane_alife_any = 0x0;
#if defined(GREEDY_SCHEDULE)
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:acquire_work_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      double dtemp_1 = sqrt(vx1.x[ii] + vy.x[ii]);
      double dtemp_2 = log(dtemp_1);
      if (dtemp_1 > 1.0) 
	dtemp_1 = dtemp_2;
      if (lane_alife.x[ii]) {
	if (!lane_acquire_work.x[ii])
	  vy.x[ii] = dtemp_1;
        k.x[ii]++;
        if (!(k.x[ii] < kmax.x[ii]))
          lane_acquire_work.x[ii] = 0x1;
        if (lane_acquire_work.x[ii]) {
          y[idx.x[ii]] = vy.x[ii];
          acquire_work_any = 0x1;
        }
      }
    }
    if (acquire_work_any)
#pragma novector
      for (int32_t ii=0; ii<VL; ii++)
	if (lane_alife.x[ii] && lane_acquire_work.x[ii])
	  idx.x[ii] = icurrent++;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:lane_alife_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      if (lane_alife.x[ii]) {
        if (idx.x[ii] < N) {
          lane_alife_any = 0x1;
        } else {
          lane_alife.x[ii] = 0x0;
        }
      }
    }
#else
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:acquire_work_any) reduction(|:lane_alife_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      double dtemp_1 = sqrt(vx1.x[ii] + vy.x[ii]);
      double dtemp_2 = log(dtemp_1);
      if (dtemp_1 > 1.0) 
	dtemp_1 = dtemp_2;
      if (lane_alife.x[ii]) {
	if (!lane_acquire_work.x[ii])
	  vy.x[ii] = dtemp_1;
        k.x[ii]++;
        if (!(k.x[ii] < kmax.x[ii]))
          lane_acquire_work.x[ii] = 0x1;
	int32_t itemp_1 = idx.x[ii];
        if (lane_acquire_work.x[ii]) {
          y[itemp_1] = vy.x[ii];
          itemp_1 += VL;
          acquire_work_any = 0x1;
        }
        if (itemp_1 < N) {
          lane_alife_any = 0x1;
        } else {
          lane_alife.x[ii] = 0x0;
        }
	idx.x[ii] = itemp_1;
      }
    }
#endif
  }

#elif defined(DYNAMIC_MANUALVEC)

#pragma message ("......Compile DYNAMIC_MANUALVEC version......")

  vec_int4_t lane_alife, lane_acquire_work, acquire_work;
  vec_int4_t idx, k, kmax;
  vec_real8_t vx1, vx2, vy, dtemp;
  vec_mask8_t m0, m1, m2;
  int32_t acquire_work_any, lane_alife_any;
#if defined(GREEDY_SCHEDULE)
  int32_t icurrent = VL;
#endif

  // prologue
  lane_alife_any = 0x1;
  acquire_work_any = 0x1;
#if defined(VECTORIZE)
#pragma omp simd safelen(VL)
#endif
  for (int32_t ii=0; ii<VL; ii++) {
    lane_alife.x[ii] = 0x1;
    lane_acquire_work.x[ii] = 0x1;
    idx.x[ii] = ii;
    m0.x[ii] = (lane_alife.x[ii] && lane_acquire_work.x[ii] ? 0x1 : 0x0);
  }

  // compute + epilogue (acquire new work)
  while (lane_alife_any) {
    if (acquire_work_any) {
#if defined(VECTORIZE)
#pragma omp simd safelen(VL)
#endif
      for (int32_t ii=0; ii<VL; ii++) {
	if (m0.x[ii]) {
	  vx1.x[ii] = x1[idx.x[ii]];
	  vx2.x[ii] = x2[idx.x[ii]];
	  vy.x[ii] = 0.0;
	  k.x[ii] = 0;
	  kmax.x[ii] = (int32_t)(D * vx2.x[ii]);
	}
       	lane_acquire_work.x[ii] = ((lane_acquire_work.x[ii]) && (k.x[ii] >= kmax.x[ii]) ? 0x1 : 0x0);
      }
    }
#if defined(VECTORIZE)
#pragma omp simd safelen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      double dtemp_1 = sqrt(vx1.x[ii] + vy.x[ii]);
      double dtemp_2 = log(dtemp_1);
      if (lane_alife.x[ii] && !lane_acquire_work.x[ii]) {
	if (dtemp_1 > 1.0) 
	  dtemp_1 = dtemp_2;
	vy.x[ii] = dtemp_1;
      }
    }
    acquire_work_any = 0x0;
    lane_alife_any = 0x0;
#if defined(GREEDY_SCHEDULE)
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:acquire_work_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      k.x[ii]++;
      if (lane_acquire_work.x[ii] || !(kmax.x[ii] > k.x[ii])) {
	lane_acquire_work.x[ii] = 0x1;
	acquire_work_any = 0x1;
      }
      m0.x[ii] = lane_alife.x[ii] && lane_acquire_work.x[ii];
      if (m0.x[ii])
	y[idx.x[ii]] = vy.x[ii];
    }
    if (acquire_work_any)
      for (int32_t ii=0; ii<VL; ii++)
	if (m0.x[ii])
	  idx.x[ii] = icurrent++;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:lane_alife_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      lane_alife.x[ii] = 0x0;
      if (idx.x[ii] < N) {
	lane_alife.x[ii] = 0x1;
	lane_alife_any = 0x1;
      } else {
	m0.x[ii] = 0x0;
      }
    }
#else
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma omp simd safelen(VL)
#else
#pragma omp simd safelen(VL) reduction(|:acquire_work_any) reduction(|:lane_alife_any)
#endif
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      k.x[ii]++;
      if (lane_acquire_work.x[ii] || !(kmax.x[ii] > k.x[ii])) {
	lane_acquire_work.x[ii] = 0x1;
	acquire_work_any = 0x1;
      }
      m0.x[ii] = lane_alife.x[ii] && lane_acquire_work.x[ii];
      if (m0.x[ii]) {
	y[idx.x[ii]] = vy.x[ii];
	idx.x[ii] += VL;
      }
      lane_alife.x[ii] = 0x0;
      if (idx.x[ii] < N) {
	lane_alife.x[ii] = 0x1;
	lane_alife_any = 0x1;
      } else {
	m0.x[ii] = 0x0;
      }
    }
#endif
  }

#elif defined(STATIC_INTRINSICS)

#pragma message ("......Compile STATIC_INTRINSICS version......")

  __MXXd vx1, vx2, vy, dtemp;
  __MXXi idx, k, kmax;
  __MMASK8 m0, m1, m2;

#if defined(__MIC__) || defined(__AVX512F__)
  idx = SIMD_SET_INT32(0,1,2,3,4,5,6,7,N,N,N,N,N,N,N,N);
#elif defined(__AVX2__)
  idx = SIMD_SET_INT32(0,1,2,3,N,N,N,N);
#endif
  for (int32_t i=0; i<N; i+=VL) {
    m0 = SIMD_CMPLT_INT32(idx, SIMD_SET1_INT32(N));
    vx1 = SIMD_MASK_LOAD_REAL64(vx1, m0, &x1[i]);
    vx2 = SIMD_MASK_LOAD_REAL64(vx2, m0, &x2[i]);
    k = SIMD_SET1_INT32(0);
    dtemp = SIMD_MUL_REAL64(SIMD_SET1_REAL64((double)D), vx2);
    kmax = SIMD_CVT_REAL64_TO_INT32(dtemp);
    m1 = SIMD_AND_MASK(m0, SIMD_CMPLT_INT32(k, kmax));
    vy = SIMD_SET1_REAL64(0.0);
    while (SIMD_MASK2INT(m1)) {
      vy = SIMD_MASK_SQRT_REAL64(vy, m1, SIMD_ADD_REAL64(vx1, vy));
      m2 = SIMD_AND_MASK(m1, SIMD_CMPGT_REAL64(vy, SIMD_SET1_REAL64(1.0)));
      if (SIMD_MASK2INT(m2))
	vy = SIMD_MASK_LOG_REAL64(vy, m2, vy);
      k = SIMD_ADD_INT32(k, SIMD_SET1_INT32(1));
      m1 = SIMD_AND_MASK(m1, SIMD_CMPLT_INT32(k, kmax));
    }
    SIMD_MASK_STORE_REAL64(&y[i], m0, vy);
    SIMD_ADD_INT32(idx, SIMD_SET1_INT32(VL));
  }

#elif defined(DYNAMIC_INTRINSICS)

#pragma message ("......Compile DYNAMIC_INTRINSICS version......")

  __MMASK8 lane_alife, lane_acquire_work, acquire_work;
  __MMASK8 m0, m1, m2;
  __MXXi idx, k, kmax;
  __MXXd vx1, vx2, vy, dtemp;
#if defined(GREEDY_SCHEDULE)
  int32_t current_idx = VL;
#endif

#if defined(__MIC__) || defined(__AVX512F__)
  lane_alife = SIMD_INT2MASK(0xFF);
  lane_acquire_work = SIMD_INT2MASK(0xFF);
  acquire_work = SIMD_INT2MASK(0xFF);
  idx = SIMD_SET_INT32(0,1,2,3,4,5,6,7,N,N,N,N,N,N,N,N);
#elif defined(__AVX2__)
  lane_alife = SIMD_SET1_MASK(0xFFFFFFFFFFFFFFFF);
  lane_acquire_work = SIMD_SET1_MASK(0xFFFFFFFFFFFFFFFF);
  acquire_work = SIMD_SET1_MASK(0xFFFFFFFFFFFFFFFF);
  idx = SIMD_SET_INT32(0,1,2,3,N,N,N,N);
#endif
  m0 = SIMD_AND_MASK(lane_alife, lane_acquire_work);
  while (SIMD_MASK2INT(lane_alife)) {
    if (SIMD_MASK2INT(m0)) { // acquire new work if any SIMD lane needs to
      k = SIMD_MASK_MOV_INT32(k, m0, SIMD_SET1_INT32(0));
      vy = SIMD_MASK_MOV_REAL64(vy, m0, SIMD_SET1_REAL64(0.0));
      vx1 = SIMD_MASK_GATHER_REAL64(vx1, m0, idx, x1);
      vx2 = SIMD_MASK_GATHER_REAL64(vx2, m0, idx, x2);
      dtemp = SIMD_MUL_REAL64(SIMD_SET1_REAL64((double)D), vx2);
      kmax = SIMD_MASK_CVT_REAL64_TO_INT32(kmax, m0, dtemp);
      lane_acquire_work = SIMD_AND_MASK(lane_acquire_work, SIMD_CMPGE_INT32(k, kmax));
    }
    { // compute
      dtemp = SIMD_SQRT_REAL64(SIMD_ADD_REAL64(vx1, vy));
      m1 = SIMD_CMPGT_REAL64(dtemp, SIMD_SET1_REAL64(1.0));
      dtemp = SIMD_MASK_LOG_REAL64(dtemp, m1, dtemp);
      m2 = SIMD_AND_MASK(lane_alife, SIMD_NOT_MASK(lane_acquire_work));
      vy = SIMD_MASK_MOV_REAL64(vy, m2, dtemp);
    }
    { // prepare next iteration + "do we need new work on any SIMD lane?"
      k = SIMD_ADD_INT32(k, SIMD_SET1_INT32(1));
      lane_acquire_work = SIMD_OR_MASK(lane_acquire_work, SIMD_CMPGE_INT32(k, kmax));
      m0 = SIMD_AND_MASK(lane_alife, lane_acquire_work);
      SIMD_MASK_SCATTER_REAL64(y, m0, idx, vy);
#if defined(GREEDY_SCHEDULE)
#if defined(__MIC__) || defined(__AVX512F__)
      idx = SIMD_MASK_MOV_INT32(idx, m0, SIMD_EXPAND_INT32(SIMD_ADD_INT32(SIMD_SET1_INT32(current_idx), SIMD_SET_INT32(0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0)), m0));
      current_idx += _mm_countbits_32(SIMD_MASK2INT(m0));
#elif defined(__AVX2__)
      idx = SIMD_MASK_MOV_INT32(idx, m0, SIMD_EXPAND_INT32(SIMD_ADD_INT32(SIMD_SET1_INT32(current_idx), SIMD_SET_INT32(0,1,2,3,0,0,0,0)), m0));
      current_idx += _mm_countbits_32(SIMD_MASK2INT(m0));
#endif
#else
      idx = SIMD_MASK_ADD_INT32(idx, m0, idx, SIMD_SET1_INT32(VL));
#endif
      lane_alife = SIMD_CMPLT_INT32(idx, SIMD_SET1_INT32(N));
      m0 = SIMD_AND_MASK(lane_alife, lane_acquire_work); 
    }
  }

#else

#pragma message ("......Compile STANDARD version......")

#if defined(VECTORIZE)
#pragma omp simd safelen(VL)
#endif
  for (int32_t i=0; i<N; i++) {
    int32_t k = 0;
    int32_t kmax = (int32_t)(D * x2[i]);
    double temp_y = 0.0;
    double temp_x1 = x1[i];
    while (k < kmax) {
      temp_y = sqrt(temp_x1 + temp_y);
      if (temp_y > 1.0) temp_y = log(temp_y);
      k++;
    }
    y[i] = temp_y;
  }

#endif

  time = omp_get_wtime() - time;

  printf("elapsed time: %.6lfms\n", time * 1.0E3);

  for (int32_t i=1; i<=16; i++)
    printf("%.6lf ", y[N-i]);
  printf("\n");

#if defined(CHECK_RESULTS)

  double *yref = (double *) _mm_malloc(N * sizeof(double), ALIGNMENT);

  // compare against no-vec case
#pragma novector
  for (int32_t i=0; i<N; i++) {
    int32_t k = 0;
    int32_t kmax = (int32_t)(D * x2[i]);
    double temp_y = 0.0;
    double temp_x1 = x1[i];
#pragma novector
    while (k < kmax) {
      temp_y = sqrt(temp_x1 + temp_y);
      if (temp_y > 1.0) temp_y = log(temp_y);
      k++;
    }
    yref[i] = temp_y;
  }

  double dev = 0.0;

  for (int32_t i=0; i<N; i++)
    dev += (y[i] - yref[i]) * (y[i] - yref[i]);
  dev = sqrt(dev);

  printf("deviation: %.6lf\n", dev);

  _mm_free(yref);

#endif

  _mm_free(x1);
  _mm_free(x2);
  _mm_free(y);
  
  return 0;

}
