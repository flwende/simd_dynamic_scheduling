// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

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
      if (SIMD_MASK2INT(m1))
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
