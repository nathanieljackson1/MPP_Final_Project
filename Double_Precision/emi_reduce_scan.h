#ifndef EMI_REDUCE_SCAN_H
#define EMI_REDUCE_SCAN_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void reduceMin(double *d_in, unsigned size, double *h_result);

void reduceMax(double *d_in, unsigned size, double *h_result);

#ifdef __cplusplus
}
#endif

#endif // EMI_REDUCE_SCAN_H
