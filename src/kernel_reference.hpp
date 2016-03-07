// Copyright (c) 2015 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if defined(VECTORIZE)
#pragma omp simd simdlen(VL)
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
