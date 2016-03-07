// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

  vec_real8_t vx1, vx2, vy;
  vec_mask8_t m0;
  vec_int4_t k, kmax;

  for (int32_t i=0; i<N; i+=VL) {
    // prologue
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      int32_t temp_m0 = 0x0;
      if ((i + ii) < N)	{
	double temp_x1 = x1[i + ii];
	double temp_x2 = x2[i + ii];
	int32_t temp_kmax = (int32_t)(D * temp_x2);
	if (0 < temp_kmax) {
	  temp_m0 = 0x1;
	  k.x[ii] = 0;
	  kmax.x[ii] = temp_kmax;
	  vx1.x[ii] = temp_x1;
	  vx2.x[ii] = temp_x2;
	}
      }
      m0.x[ii] = temp_m0;
      vy.x[ii] = 0.0;
    }
    // compute
    int32_t continue_loop = 0x1;
    while (continue_loop) {
      continue_loop = 0x0;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
      for (int32_t ii=0; ii<VL; ii++) {
	int32_t temp_m0 = m0.x[ii];
	double temp_y = vy.x[ii];
	if (temp_m0) {
	  temp_y = sqrt(vx1.x[ii] + temp_y);
	  double temp_1 = log(temp_y);
	  if (temp_y > 1.0) 
	    temp_y = temp_1;
	  if (++k.x[ii] < kmax.x[ii])
	    continue_loop = 0x1;
	  else
	    temp_m0 = 0x0;
        }
	vy.x[ii] = temp_y;
	m0.x[ii] = temp_m0;
      }
    }
    // epilogue
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif                                                                                            
    for (int32_t ii=0; ii<VL; ii++)
      if ((i+ii) < N) 
	y[i+ii] = vy.x[ii];
  }
