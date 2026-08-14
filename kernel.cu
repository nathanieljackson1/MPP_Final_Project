
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_CONST_PARAM 2048
__constant__ double R_const[MAX_CONST_PARAM];
__constant__ double fsw_const[MAX_CONST_PARAM];

// Helper Functions (performed in kernels so operating on GPU devices) ////////////////////
__device__ double computeTransferMagnitude(double L, double C, double R, double f)
{
    double omega = 2.0 * M_PI * f;
    double real_part = 1.0 - omega * omega * L * C;
    double imag_part = omega * R * C;
    return 1.0 / sqrt(real_part * real_part + imag_part * imag_part);
}

__device__ void decodeIndex(unsigned idx, unsigned numL, unsigned numC, unsigned numR, unsigned* l_idx, unsigned* c_idx, unsigned* r_idx, unsigned* f_idx)
{
    unsigned i = idx;
    *l_idx = i % numL;   i /= numL;
    *c_idx = i % numC;   i /= numC;
    *r_idx = i % numR;   i /= numR;
    *f_idx = i;
}

__device__ void evaluateOne(double L, double C, double R, double f_sw, unsigned numHarmonics, double* worstOut, double* thdOut)
{
    double worst = 1e30; // will always be larger than real values
    double thd_num = 0.0, thd_den = 1.0;

    for(unsigned h = 1; h <= numHarmonics; h++) {
        double f = f_sw * h;
        double H_mag = computeTransferMagnitude(L, C, R, f);
        double atten_dB = -20.0 * log10(H_mag);

        if(atten_dB < worst) worst = atten_dB;
        if(h == 1) thd_den = H_mag;
        else thd_num += H_mag * H_mag;
    }

    *worstOut = worst;
    *thdOut = sqrt(thd_num) / thd_den;
}

__device__ void computeGroupRange(unsigned blockFirstIdx, unsigned blockLastIdx, unsigned numL, unsigned numC, unsigned* groupFirst, unsigned* groupSpan)
{
    unsigned LCper = numL * numC;
    unsigned gFirst = blockFirstIdx / LCper;
    unsigned gLast  = blockLastIdx  / LCper;
    *groupFirst = gFirst;
    *groupSpan = gLast - gFirst + 1;
}

// Kernel Version 1: naive /////////////////////////
__global__ void evaluateFilterKernel_naive(const double* __restrict__ L_vals, const double* __restrict__ C_vals, const double* __restrict__ R_vals, const double* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, double* worstAttenuation, double* thd)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned totalCombinations = numL * numC * numR * numFsw;
    if(idx >= totalCombinations) return;

    unsigned l_idx, c_idx, r_idx, f_idx;
    decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

    double L = L_vals[l_idx];
    double C = C_vals[c_idx];
    double R = R_vals[r_idx];
    double f_sw = fsw_vals[f_idx];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}

// Kernel Version 2: shared memory /////////////////////////
__global__ void evaluateFilterKernel_shared(const double* __restrict__ L_vals, const double* __restrict__ C_vals, const double* __restrict__ R_vals, const double* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, double* worstAttenuation, double* thd)
{
    extern __shared__ double s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned blockFirstIdx = blockIdx.x * blockDim.x;
    unsigned blockLastIdx = min(blockFirstIdx + blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    double* s_R = s_group;
    double* s_fsw = s_group + groupSpan;

    for(unsigned i = threadIdx.x; i < groupSpan; i += blockDim.x) {
        unsigned g = groupFirst + i;
        s_R[i] = R_vals[g % numR];
        s_fsw[i] = fsw_vals[g / numR];
    }
    __syncthreads();

    unsigned idx = blockFirstIdx + threadIdx.x;
    if(idx >= totalCombinations) return;

    unsigned l_idx, c_idx, r_idx, f_idx;
    decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

    unsigned slot = (r_idx + numR * f_idx) - groupFirst;

    double L = L_vals[l_idx];
    double C = C_vals[c_idx];
    double R = s_R[slot];
    double f_sw = s_fsw[slot];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}

// Kernel Version 3: divergence-minimized /////////////////////////
__global__ void evaluateFilterKernel_divMin(const double* __restrict__ L_vals, const double* __restrict__ C_vals, const double* __restrict__ R_vals, const double* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, double* worstAttenuation, double* thd)
{
    extern __shared__ double s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned blockFirstIdx = blockIdx.x * blockDim.x;
    unsigned blockLastIdx = min(blockFirstIdx + blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    double* s_R = s_group;
    double* s_fsw = s_group + groupSpan;

    for(unsigned i = threadIdx.x; i < groupSpan; i += blockDim.x) {
        unsigned g = groupFirst + i;
        s_R[i] = R_vals[g % numR];
        s_fsw[i] = fsw_vals[g / numR];
    }
    __syncthreads();

    unsigned numFullWarpBlocks = totalCombinations / blockDim.x;
    unsigned idx = blockFirstIdx + threadIdx.x;

    if(blockIdx.x < numFullWarpBlocks) {
        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

        unsigned slot = (r_idx + numR * f_idx) - groupFirst;
        double L = L_vals[l_idx];
        double C = C_vals[c_idx];
        double R = s_R[slot];
        double f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    } else if(idx < totalCombinations) {
        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

        unsigned slot = (r_idx + numR * f_idx) - groupFirst;
        double L = L_vals[l_idx];
        double C = C_vals[c_idx];
        double R = s_R[slot];
        double f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    }
}

// Version 4: thread coarsened /////////////////////////
__global__ void evaluateFilterKernel_coarsened(
    const double* __restrict__ L_vals, const double* __restrict__ C_vals,
    const double* __restrict__ R_vals, const double* __restrict__ fsw_vals,
    unsigned numL, unsigned numC, unsigned numR, unsigned numFsw,
    unsigned numHarmonics, unsigned coarseFactor,
    double* worstAttenuation, double* thd)
{
    extern __shared__ double s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned segment = coarseFactor * blockDim.x * blockIdx.x;
    unsigned blockFirstIdx = segment;
    unsigned blockLastIdx = min(segment + coarseFactor * blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    double* s_R = s_group;
    double* s_fsw = s_group + groupSpan;

    for(unsigned i = threadIdx.x; i < groupSpan; i += blockDim.x) {
        unsigned g = groupFirst + i;
        s_R[i] = R_vals[g % numR];
        s_fsw[i] = fsw_vals[g / numR];
    }
    __syncthreads();

    for(unsigned c = 0; c < coarseFactor; c++) {
        unsigned idx = segment + c * blockDim.x + threadIdx.x;
        if(idx >= totalCombinations) continue;

        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

        unsigned slot = (r_idx + numR * f_idx) - groupFirst;
        double L = L_vals[l_idx];
        double C = C_vals[c_idx];
        double R = s_R[slot];
        double f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    }
}

// Version 5: constant memory /////////////////////////
__global__ void evaluateFilterKernel_constant(
    const double* __restrict__ L_vals, const double* __restrict__ C_vals,
    unsigned numL, unsigned numC, unsigned numR, unsigned numFsw,
    unsigned numHarmonics,
    double* worstAttenuation, double* thd)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned totalCombinations = numL * numC * numR * numFsw;
    if(idx >= totalCombinations) return;

    unsigned l_idx, c_idx, r_idx, f_idx;
    decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

    double L = L_vals[l_idx];
    double C = C_vals[c_idx];
    double R = R_const[r_idx];
    double f_sw = fsw_const[f_idx];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}