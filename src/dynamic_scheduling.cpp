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
//#define GREEDY_SCHEDULE // use vector expand operation for dynamic scheduling
#define CHECK_RESULTS // compare against non-SIMD version
#define OPTIMIZED_KERNELS

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

#if defined(OPTIMIZED_KERNELS)
#include "kernel_static_manual_opt.hpp"
#else
#include "kernel_static_manual.hpp"
#endif

#elif defined(DYNAMIC_MANUALVEC)

#pragma message ("......Compile DYNAMIC_MANUALVEC version......")

#if defined(OPTIMIZED_KERNELS)
#include "kernel_dynamic_manual_opt.hpp"
#else
#include "kernel_dynamic_manual.hpp"
#endif

#elif defined(STATIC_INTRINSICS)

#pragma message ("......Compile STATIC_INTRINSICS version......")

#include "kernel_static_intrinsics.hpp"

#elif defined(DYNAMIC_INTRINSICS)

#pragma message ("......Compile DYNAMIC_INTRINSICS version......")

#include "kernel_dynamic_intrinsics.hpp"

#else

#pragma message ("......Compile REFERENCE version......")

#include "kernel_reference.hpp"

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
