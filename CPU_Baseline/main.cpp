#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "support_cpu.h"
#include "kernel_cpu.h"

// Runs one full double sweep and report timing
static double runDoubleSweep(unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, std::vector<double>& worstAtten_out, std::vector<double>& thd_out)
{
    Timer timer;
    unsigned total = numL * numC * numR * numFsw;

    std::vector<double> L_h = dp::initParamVector(numL, 1e-6, 1e-3);
    std::vector<double> C_h = dp::initParamVector(numC, 1e-9, 1e-5);
    std::vector<double> R_h = dp::initParamVector(numR, 0.01, 10.0);
    std::vector<double> Fsw_h = dp::initParamVector(numFsw, 1e4, 2e5);

    worstAtten_out.resize(total);
    thd_out.resize(total);

    startTime(&timer);
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)total; idx++) {
        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex((unsigned)idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);
        dp::evaluateOne(L_h[l_idx], C_h[c_idx], R_h[r_idx], Fsw_h[f_idx],
            NUM_HARMONICS, &worstAtten_out[idx], &thd_out[idx]);
    }
    stopTime(&timer);

    return elapsedTime(timer);
}

// Runs one full float sweep and report timing
static double runFloatSweep(unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, std::vector<float>& worstAtten_out, std::vector<float>& thd_out)
{
    Timer timer;
    unsigned total = numL * numC * numR * numFsw;

    std::vector<float> L_h = sp::initParamVector(numL, 1e-6, 1e-3);
    std::vector<float> C_h = sp::initParamVector(numC, 1e-9, 1e-5);
    std::vector<float> R_h = sp::initParamVector(numR, 0.01, 10.0);
    std::vector<float> Fsw_h = sp::initParamVector(numFsw, 1e4, 2e5);

    worstAtten_out.resize(total);
    thd_out.resize(total);

    startTime(&timer);
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)total; idx++) {
        unsigned l_idx, c_idx, r_idx, f_idx;
        decodeIndex((unsigned)idx, numL, numC, numR, &l_idx, &c_idx, &r_idx, &f_idx);
        sp::evaluateOne(L_h[l_idx], C_h[c_idx], R_h[r_idx], Fsw_h[f_idx],
            NUM_HARMONICS, &worstAtten_out[idx], &thd_out[idx]);
    }
    stopTime(&timer);

    return elapsedTime(timer);
}

// absolute and relative correctness check
static void compareResults(const std::vector<double>& refWorst, const std::vector<double>& refThd, const std::vector<float>& gotWorst, const std::vector<float>& gotThd)
{
    const double ABS_TOL = 1e-2;
    const double REL_TOL = 5e-3;
    unsigned total = (unsigned)refWorst.size();
    unsigned mismatches = 0;
    unsigned printed = 0;

    for (unsigned i = 0; i < total; i++) {
        double errWorst = fabs(refWorst[i] - (double)gotWorst[i]);
        double errThd = fabs(refThd[i] - (double)gotThd[i]);

        bool worstMismatch = (errWorst > ABS_TOL) && (errWorst / fabs(refWorst[i]) > REL_TOL);
        bool thdMismatch = (errThd > ABS_TOL) && (errThd / fabs(refThd[i]) > REL_TOL);

        if (worstMismatch || thdMismatch) {
            mismatches++;
            if (printed < 5) {
                printf("    Mismatch at idx %u: worst ref=%f got=%f | thd ref=%f got=%f\n",
                    i, refWorst[i], (double)gotWorst[i], refThd[i], (double)gotThd[i]);
                printed++;
            }
        }
    }

    if (mismatches == 0) {
        printf("    TEST PASSED (float vs. double reference, ABS_TOL=%.0e, REL_TOL=%.0e)\n", ABS_TOL, REL_TOL);
    } else {
        printf("    TEST FAILED: %u / %u combinations mismatched\n", mismatches, total);
    }
}

// single problem size at both double and single precision
static void runOneConfig(unsigned numL, unsigned numC, unsigned numR, unsigned numFsw)
{
    unsigned total = numL * numC * numR * numFsw;

    printf("\n================================================================\n");
    printf("Sweep size = %u (L:%u x C:%u x R:%u x Fsw:%u)\n", total, numL, numC, numR, numFsw);
    printf("================================================================\n");

#ifdef _OPENMP
    printf("(OpenMP enabled, %d threads)\n", omp_get_max_threads());
#else
    printf("(serial, single-threaded)\n");
#endif

    // Double precision pass
    std::vector<double> worstD, thdD;
    printf("\n[DOUBLE] Computing..."); fflush(stdout);
    double tDouble = runDoubleSweep(numL, numC, numR, numFsw, worstD, thdD);
    printf(" %f s\n", tDouble);

    double worstGlobalD = *std::min_element(worstD.begin(), worstD.end());
    double thdGlobalD = *std::max_element(thdD.begin(), thdD.end());
    printf("[DOUBLE] Global worst-case attenuation = %f dB, THD = %f\n", worstGlobalD, thdGlobalD);
    printf("[DOUBLE] Throughput: %.2f million combinations/sec\n", (total / tDouble) / 1e6);

    // Single precision pass
    std::vector<float> worstF, thdF;
    printf("\n[FLOAT]  Computing..."); fflush(stdout);
    double tFloat = runFloatSweep(numL, numC, numR, numFsw, worstF, thdF);
    printf(" %f s\n", tFloat);

    float worstGlobalF = *std::min_element(worstF.begin(), worstF.end());
    float thdGlobalF = *std::max_element(thdF.begin(), thdF.end());
    printf("[FLOAT]  Global worst-case attenuation = %f dB, THD = %f\n", worstGlobalF, thdGlobalF);
    printf("[FLOAT]  Throughput: %.2f million combinations/sec\n", (total / tFloat) / 1e6);

    // Comparison
    printf("\n[COMPARE] Verifying float results against double reference...\n");
    compareResults(worstD, thdD, worstF, thdF);

    printf("\n[SUMMARY] size=%u  double=%.4fs  float=%.4fs  speedup(float/double)=%.2fx\n",
        total, tDouble, tFloat, tDouble / tFloat);
}

// main cpu test implementation
int main(int argc, char* argv[])
{
    if (argc == 1) {
        printf("No arguments given: running the 3 recommended GPU-matching sizes:\n");
        printf("  32x78x283x283, 32x78x448x448, 32x78x633x633\n");

        runOneConfig(32, 78, 283, 283);
        runOneConfig(32, 78, 448, 448);
        runOneConfig(32, 78, 633, 633);

    } else if (argc == 5) {
        unsigned numL   = (unsigned)atoi(argv[1]);
        unsigned numC   = (unsigned)atoi(argv[2]);
        unsigned numR   = (unsigned)atoi(argv[3]);
        unsigned numFsw = (unsigned)atoi(argv[4]);
        runOneConfig(numL, numC, numR, numFsw);

    } else {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./cpu_baseline                              # Run 3 default sizes"
               "\n    Usage: ./cpu_baseline <numL> <numC> <numR> <numFsw>  # Run one custom size"
               "\n");
        exit(0);
    }

    return 0;
}
