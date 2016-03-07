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
      m0.x[ii] = 0x0;
      if ((i + ii) < N)	{
	kmax.x[ii] = (int32_t)(D * x2[i + ii]);
	if (0 < kmax.x[ii]) {
	  m0.x[ii] = 0x1;
	  k.x[ii] = 0;
	  vx1.x[ii] = x1[i + ii];
	  vx2.x[ii] = x2[i + ii];
	}
      }
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
	if (m0.x[ii]) {
	  vy.x[ii] = sqrt(vx1.x[ii] + vy.x[ii]);
	  if (vy.x[ii] > 1.0) 
	    vy.x[ii] = log(vy.x[ii]);
	  if (++k.x[ii] < kmax.x[ii])
	    continue_loop = 0x1;
	  else
	    m0.x[ii] = 0x0;
        }
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
