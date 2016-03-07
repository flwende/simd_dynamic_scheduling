  vec_int4_t lane_alife, lane_acquire_work, acquire_work;
  vec_int4_t idx, k, kmax;
  vec_real8_t vx1, vx2, vy, dtemp;
  int32_t acquire_work_any, lane_alife_any;
#if defined(GREEDY_SCHEDULE)
  int32_t icurrent = VL;
#endif

  // prologue
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
  for (int32_t ii=0; ii<VL; ii++) {
    if (ii < N)
      {
	lane_alife.x[ii] = 0x1;
	lane_acquire_work.x[ii] = 0x1;
	idx.x[ii] = ii;
      }
    else
      {
	lane_alife.x[ii] = 0x0;
	lane_acquire_work.x[ii] = 0x0;
      }
  }

  // compute + epilogue (acquire new work)
  lane_alife_any = 0x1;
  acquire_work_any = 0x1;
  while (lane_alife_any) {
    lane_alife_any = 0x0;
    if (acquire_work_any) {
      acquire_work_any = 0x0;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
      for (int32_t ii=0; ii<VL; ii++) {
	if (lane_acquire_work.x[ii]) {
	  vx1.x[ii] = x1[idx.x[ii]];
	  vx2.x[ii] = x2[idx.x[ii]];
	  kmax.x[ii] = (int32_t)(D * vx2.x[ii]);
	  if (0 < kmax.x[ii]) {
	    lane_acquire_work.x[ii] = 0x0;
	    k.x[ii] = 0;
	  }
	  vy.x[ii] = 0.0;
	}
      }
    }
    acquire_work_any = 0x0;
    lane_alife_any = 0x0;
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      if (lane_alife.x[ii] && !lane_acquire_work.x[ii]) {
	vy.x[ii] = sqrt(vx1.x[ii] + vy.x[ii]);
	if (vy.x[ii] > 1.0) 
	  vy.x[ii] = log(vy.x[ii]);
	if (++k.x[ii] >= kmax.x[ii])
	  lane_acquire_work.x[ii] = 0x1;
      }
#if defined(GREEDY_SCHEDULE)
      if (lane_acquire_work.x[ii])
	y[idx.x[ii]] = vy.x[ii];
    }
    for (int32_t ii=0; ii<VL; ii++) {
      if (lane_acquire_work.x[ii])
	idx.x[ii] = icurrent++;
    }
#if defined(VECTORIZE)
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#pragma omp simd simdlen(VL)
#endif
    for (int32_t ii=0; ii<VL; ii++) {
      if (lane_acquire_work.x[ii]) {
	if (idx.x[ii] >= N) {
	  lane_acquire_work.x[ii] = 0x0;
	  lane_alife.x[ii] = 0x0;
	} else {
	  acquire_work_any = 0x1;
	}
      }
#else
      if (lane_acquire_work.x[ii]) {
	y[idx.x[ii]] = vy.x[ii];
	idx.x[ii] += VL;
	if (idx.x[ii] >= N) {
	  lane_acquire_work.x[ii] = 0x0;
	  lane_alife.x[ii] = 0x0;
	} else {
	  acquire_work_any = 0x1;
	}
      }
#endif
      if (lane_alife.x[ii])
	lane_alife_any = 0x1;
    }
  }  
