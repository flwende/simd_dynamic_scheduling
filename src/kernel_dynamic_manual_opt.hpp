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
	int32_t temp_idx = idx.x[ii];
	int32_t temp_lane_acquire_work = lane_acquire_work.x[ii];
	if (temp_lane_acquire_work) {
	  double temp_x1 = x1[temp_idx];
	  double temp_x2 = x2[temp_idx];
	  int32_t temp_kmax = (int32_t)(D * temp_x2);
	  if (0 < temp_kmax) {
	    temp_lane_acquire_work = 0x0;
	    k.x[ii] = 0;
	    kmax.x[ii] = temp_kmax;
	    vx1.x[ii] = temp_x1;
	    vx2.x[ii] = temp_x2;
	  }
	  lane_acquire_work.x[ii] = temp_lane_acquire_work;
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
      double temp_y = vy.x[ii];
      int32_t temp_lane_alife = lane_alife.x[ii];
      int32_t temp_lane_acquire_work = lane_acquire_work.x[ii];
      if (temp_lane_alife && !temp_lane_acquire_work) {
	temp_y = sqrt(vx1.x[ii] + temp_y);
	double temp_1 = log(temp_y);
	if (temp_y > 1.0) 
	  temp_y = temp_1;
	if (++k.x[ii] >= kmax.x[ii])
	  temp_lane_acquire_work = 0x1;
	vy.x[ii] = temp_y;
      }
#if defined(GREEDY_SCHEDULE)
      if (temp_lane_acquire_work)
	y[idx.x[ii]] = vy.x[ii];
      lane_acquire_work.x[ii] = temp_lane_acquire_work;
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
      int32_t temp_lane_alife = lane_alife.x[ii];
      int32_t temp_lane_acquire_work = lane_acquire_work.x[ii];
      if (temp_lane_acquire_work) {
	if (idx.x[ii] >= N) {
	  temp_lane_acquire_work = 0x0;
	  temp_lane_alife = 0x0;
	} else {
	  acquire_work_any = 0x1;
	}
      }
#else
      int32_t temp_idx = idx.x[ii];
      if (temp_lane_acquire_work) {
	y[temp_idx] = vy.x[ii];
	temp_idx += VL;
	if (temp_idx >= N) {
	  temp_lane_acquire_work = 0x0;
	  temp_lane_alife = 0x0;
	} else {
	  acquire_work_any = 0x1;
	}
	idx.x[ii] = temp_idx;
      }
#endif
      if (temp_lane_alife)
	lane_alife_any = 0x1;
      lane_acquire_work.x[ii] = temp_lane_acquire_work;
      lane_alife.x[ii] = temp_lane_alife;
    }
  }  
