
#ifndef EMI_REDUCE_SCAN_H
#define EMI_REDUCE_SCAN_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void reduceMin(float *d_in, unsigned size, float *h_result);

void reduceMax(float *d_in, unsigned size, float *h_result);

void prefixScan(float *d_out, float *d_in, unsigned size);

#ifdef __cplusplus
}
#endif

#endif // EMI_REDUCE_SCAN_H