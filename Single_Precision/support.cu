#include <math.h>
#include "support.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper Functions for project kernels
void initParamVector(float** vec_h, unsigned num, float minVal, float maxVal)
{
    *vec_h = (float*)malloc(num * sizeof(float));
    if(*vec_h == NULL) FATAL("Unable to allocate host param vector");

    double logMin = log10((double)minVal);
    double logMax = log10((double)maxVal);
    double step = (num > 1) ? (logMax - logMin) / (num - 1) : 0.0;

    for(unsigned i = 0; i < num; i++) {
        (*vec_h)[i] = (float)pow(10.0, logMin + i * step);
    }
}

static double cpuTransferMagnitude(double L, double C, double R, double f)
{
    double omega = 2.0 * M_PI * f;
    double real_part = 1.0 - omega * omega * L * C;
    double imag_part = omega * R * C;
    return 1.0 / sqrt(real_part * real_part + imag_part * imag_part);
}

static void cpuEvaluateOne(double L, double C, double R, double f_sw, unsigned numHarmonics, double* worstOut, double* thdOut)
{
    double worst = 1e30;
    double thd_num = 0.0, thd_den = 1.0;

    for(unsigned h = 1; h <= numHarmonics; h++) {
        double f = f_sw * h;
        double H_mag = cpuTransferMagnitude(L, C, R, f);
        double atten_dB = -20.0 * log10(H_mag);

        if(atten_dB < worst) worst = atten_dB;
        if(h == 1) thd_den = H_mag;
        else thd_num += H_mag * H_mag;
    }

    *worstOut = worst;
    *thdOut = sqrt(thd_num) / thd_den;
}

// correctness verification function
void verify(float* L_h, float* C_h, float* R_h, float* fsw_h, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, float* worstAtten_h, float* thd_h, unsigned totalCombinations)
{
    const double ABS_TOL = 1e-2;
    const double REL_TOL = 5e-3;
    unsigned mismatches = 0;
    unsigned printed = 0;

    for (unsigned idx = 0; idx < totalCombinations; idx++) {
        unsigned i = idx;
        unsigned l_idx = i % numL; i /= numL;
        unsigned c_idx = i % numC; i /= numC;
        unsigned r_idx = i % numR; i /= numR;
        unsigned f_idx = i;

        double refWorst, refThd;
        cpuEvaluateOne((double)L_h[l_idx], (double)C_h[c_idx],
            (double)R_h[r_idx], (double)fsw_h[f_idx],
            numHarmonics, &refWorst, &refThd);

        double errWorst = fabs(refWorst - (double)worstAtten_h[idx]);
        double errThd = fabs(refThd - (double)thd_h[idx]);

        bool worstMismatch = (errWorst > ABS_TOL) &&
            (errWorst / fabs(refWorst) > REL_TOL);
        bool thdMismatch = (errThd > ABS_TOL) &&
            (errThd / fabs(refThd) > REL_TOL);

        if (worstMismatch || thdMismatch) {
            mismatches++;
            if (printed < 5) {
                printf("\n  Mismatch at idx %u: worst ref=%f got=%f | thd ref=%f got=%f",
                    idx, refWorst, (double)worstAtten_h[idx], refThd, (double)thd_h[idx]);
                printed++;
            }
        }
    }

    if (mismatches == 0) {
        printf("\nTEST PASSED (ABS_TOL=%.0e, REL_TOL=%.0e)\n", ABS_TOL, REL_TOL);
    } else {
        printf("\nTEST FAILED: %u / %u combinations mismatched\n", mismatches, totalCombinations);
    }
}