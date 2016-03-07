// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

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
