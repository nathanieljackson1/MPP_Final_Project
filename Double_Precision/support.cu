#include <math.h>
#include "support.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void initParamVector(double** vec_h, unsigned num, double minVal, double maxVal)
{
    *vec_h = (double*)malloc(num * sizeof(double));
    if(*vec_h == NULL) FATAL("Unable to allocate host param vector");

    double logMin = log10(minVal);
    double logMax = log10(maxVal);
    double step = (num > 1) ? (logMax - logMin) / (num - 1) : 0.0;

    for(unsigned i = 0; i < num; i++) {
        (*vec_h)[i] = pow(10.0, logMin + i * step);
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

// verification function to ensure it operated correctly with tolerance
void verify(double* L_h, double* C_h, double* R_h, double* fsw_h, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, double* worstAtten_h, double* thd_h, unsigned totalCombinations)
{
    const double TOL = 1e-6;
    unsigned mismatches = 0;

    for(unsigned idx = 0; idx < totalCombinations; idx++) {
        unsigned i = idx;
        unsigned l_idx = i % numL;   i /= numL;
        unsigned c_idx = i % numC;   i /= numC;
        unsigned r_idx = i % numR;   i /= numR;
        unsigned f_idx = i;

        double refWorst, refThd;
        cpuEvaluateOne(L_h[l_idx], C_h[c_idx], R_h[r_idx], fsw_h[f_idx],
            numHarmonics, &refWorst, &refThd);

        double errWorst = fabs(refWorst - worstAtten_h[idx]);
        double errThd = fabs(refThd - thd_h[idx]);

        if(errWorst > TOL || errThd > TOL) {
            mismatches++;
            if(mismatches <= 5) {
                printf("\n    Mismatch at idx %u: worst ref=%f got=%f | thd ref=%f got=%f",
                    idx, refWorst, worstAtten_h[idx], refThd, thd_h[idx]);
            }
        }
    }

    if(mismatches == 0) {
        printf("TEST PASSED\n");
    } else {
        printf("\nTEST FAILED: %u / %u combinations mismatched\n", mismatches, totalCombinations);
    }
}
