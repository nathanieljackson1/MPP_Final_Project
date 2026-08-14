
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// single-precision pi
#define PI_F 3.14159265358979323846f

#define MAX_CONST_PARAM 2048
__constant__ float R_const[MAX_CONST_PARAM];
__constant__ float fsw_const[MAX_CONST_PARAM];

// Helper functions
__device__ float computeTransferMagnitude(float L, float C, float R, float f)
{
    double omega = 2.0 * M_PI * (double)f;
    double real_part = 1.0 - omega * omega * (double)L * (double)C;
    double imag_part = omega * (double)R * (double)C;
    return (float)(1.0 / sqrt(real_part * real_part + imag_part * imag_part));
}

__device__ void decodeIndex(unsigned idx, unsigned numL, unsigned numC, unsigned numR, unsigned* l_idx, unsigned* c_idx, unsigned* r_idx, unsigned* f_idx)
{
    unsigned i = idx;
    *l_idx = i % numL;   i /= numL;
    *c_idx = i % numC;   i /= numC;
    *r_idx = i % numR;   i /= numR;
    *f_idx = i;
}

__device__ void evaluateOne(float L, float C, float R, float f_sw, unsigned numHarmonics, float* worstOut, float* thdOut)
{
    float worst = 1e30f;
    float thd_num = 0.0f, thd_den = 1.0f;

    for(unsigned h = 1; h <= numHarmonics; h++) {
        float f = f_sw * h;
        float H_mag = computeTransferMagnitude(L, C, R, f);
        float atten_dB = -20.0f * log10f(H_mag);

        if(atten_dB < worst) worst = atten_dB;
        if(h == 1) thd_den = H_mag;
        else thd_num += H_mag * H_mag;
    }

    *worstOut = worst;
    *thdOut = sqrtf(thd_num) / thd_den;
}

__device__ void computeGroupRange(
    unsigned blockFirstIdx, unsigned blockLastIdx, unsigned numL, unsigned numC, unsigned* groupFirst, unsigned* groupSpan)
{
    unsigned LCper = numL * numC;
    unsigned gFirst = blockFirstIdx / LCper;
    unsigned gLast  = blockLastIdx  / LCper;
    *groupFirst = gFirst;
    *groupSpan = gLast - gFirst + 1;
}

// Kernel Version 1: naive
__global__ void evaluateFilterKernel_naive(const float* __restrict__ L_vals, const float* __restrict__ C_vals, const float* __restrict__ R_vals, const float* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, float* worstAttenuation, float* thd)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned totalCombinations = numL * numC * numR * numFsw;
    if(idx >= totalCombinations) return;

    unsigned l_idx, c_idx, r_idx, f_idx;
    decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

    float L = L_vals[l_idx];
    float C = C_vals[c_idx];
    float R = R_vals[r_idx];
    float f_sw = fsw_vals[f_idx];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}

// Kernel Version 2: shared memory
__global__ void evaluateFilterKernel_shared(const float* __restrict__ L_vals, const float* __restrict__ C_vals, const float* __restrict__ R_vals, const float* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, float* worstAttenuation, float* thd)
{
    extern __shared__ float s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned blockFirstIdx = blockIdx.x * blockDim.x;
    unsigned blockLastIdx = min(blockFirstIdx + blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    float* s_R = s_group;
    float* s_fsw = s_group + groupSpan;

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

    float L = L_vals[l_idx];
    float C = C_vals[c_idx];
    float R = s_R[slot];
    float f_sw = s_fsw[slot];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}

// Kernel Version 3: divergence-minimized
__global__ void evaluateFilterKernel_divMin(const float* __restrict__ L_vals, const float* __restrict__ C_vals, const float* __restrict__ R_vals, const float* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, float* worstAttenuation, float* thd)
{
    extern __shared__ float s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned blockFirstIdx = blockIdx.x * blockDim.x;
    unsigned blockLastIdx = min(blockFirstIdx + blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    float* s_R = s_group;
    float* s_fsw = s_group + groupSpan;

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
        float L = L_vals[l_idx];
        float C = C_vals[c_idx];
        float R = s_R[slot];
        float f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    } else if(idx < totalCombinations) {
        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

        unsigned slot = (r_idx + numR * f_idx) - groupFirst;
        float L = L_vals[l_idx];
        float C = C_vals[c_idx];
        float R = s_R[slot];
        float f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    }
}

// Kernel Version 4: coarsened
__global__ void evaluateFilterKernel_coarsened(const float* __restrict__ L_vals, const float* __restrict__ C_vals, const float* __restrict__ R_vals, const float* __restrict__ fsw_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, unsigned coarseFactor, float* worstAttenuation, float* thd)
{
    extern __shared__ float s_group[];

    unsigned totalCombinations = numL * numC * numR * numFsw;
    unsigned segment = coarseFactor * blockDim.x * blockIdx.x;
    unsigned blockFirstIdx = segment;
    unsigned blockLastIdx = min(segment + coarseFactor * blockDim.x - 1, totalCombinations - 1);

    unsigned groupFirst, groupSpan;
    computeGroupRange(blockFirstIdx, blockLastIdx, numL, numC, &groupFirst, &groupSpan);

    float* s_R = s_group;
    float* s_fsw = s_group + groupSpan;

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
        float L = L_vals[l_idx];
        float C = C_vals[c_idx];
        float R = s_R[slot];
        float f_sw = s_fsw[slot];

        evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
    }
}

// Kernel Version 5: constant memory
__global__ void evaluateFilterKernel_constant(const float* __restrict__ L_vals, const float* __restrict__ C_vals, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, float* worstAttenuation, float* thd)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned totalCombinations = numL * numC * numR * numFsw;
    if(idx >= totalCombinations) return;

    unsigned l_idx, c_idx, r_idx, f_idx;
    decodeIndex(idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);

    float L = L_vals[l_idx];
    float C = C_vals[c_idx];
    float R = R_const[r_idx];
    float f_sw = fsw_const[f_idx];

    evaluateOne(L, C, R, f_sw, numHarmonics, &worstAttenuation[idx], &thd[idx]);
}