#ifndef KERNEL_CPU_H
#define KERNEL_CPU_H

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const unsigned NUM_HARMONICS = 15;

// matching decode index
inline void decodeIndex(unsigned idx, unsigned numL, unsigned numC, unsigned numR, unsigned* l_idx, unsigned* c_idx, unsigned* r_idx, unsigned* f_idx)
{
    unsigned i = idx;
    *l_idx = i % numL;   i /= numL;
    *c_idx = i % numC;   i /= numC;
    *r_idx = i % numR;   i /= numR;
    *f_idx = i;
}

// double precision helper functions mirroring GPU
namespace dp {

inline double computeTransferMagnitude(double L, double C, double R, double f)
{
    double omega = 2.0 * M_PI * f;
    double real_part = 1.0 - omega * omega * L * C;
    double imag_part = omega * R * C;
    return 1.0 / sqrt(real_part * real_part + imag_part * imag_part);
}

inline void evaluateOne(double L, double C, double R, double f_sw, unsigned numHarmonics, double* worstOut, double* thdOut)
{
    double worst = 1e30;
    double thd_num = 0.0, thd_den = 1.0;

    for (unsigned h = 1; h <= numHarmonics; h++) {
        double f = f_sw * h;
        double H_mag = computeTransferMagnitude(L, C, R, f);
        double atten_dB = -20.0 * log10(H_mag);

        if (atten_dB < worst) worst = atten_dB;
        if (h == 1) thd_den = H_mag;
        else thd_num += H_mag * H_mag;
    }

    *worstOut = worst;
    *thdOut = sqrt(thd_num) / thd_den;
}

inline std::vector<double> initParamVector(unsigned num, double minVal, double maxVal)
{
    std::vector<double> vec(num);
    double logMin = log10(minVal);
    double logMax = log10(maxVal);
    double step = (num > 1) ? (logMax - logMin) / (num - 1) : 0.0;
    for (unsigned i = 0; i < num; i++) vec[i] = pow(10.0, logMin + i * step);
    return vec;
}

} // namespace dp

// single precision helper functions mirroring GPU
namespace sp {

inline float computeTransferMagnitude(float L, float C, float R, float f)
{
    double omega = 2.0 * M_PI * (double)f;
    double real_part = 1.0 - omega * omega * (double)L * (double)C;
    double imag_part = omega * (double)R * (double)C;
    return (float)(1.0 / sqrt(real_part * real_part + imag_part * imag_part));
}

inline void evaluateOne(float L, float C, float R, float f_sw, unsigned numHarmonics, float* worstOut, float* thdOut)
{
    float worst = 1e30f;
    float thd_num = 0.0f, thd_den = 1.0f;

    for (unsigned h = 1; h <= numHarmonics; h++) {
        float f = f_sw * (float)h;
        float H_mag = computeTransferMagnitude(L, C, R, f);
        float atten_dB = -20.0f * log10f(H_mag);

        if (atten_dB < worst) worst = atten_dB;
        if (h == 1) thd_den = H_mag;
        else thd_num += H_mag * H_mag;
    }

    *worstOut = worst;
    *thdOut = sqrtf(thd_num) / thd_den;
}

inline std::vector<float> initParamVector(unsigned num, double minVal, double maxVal)
{
    std::vector<float> vec(num);
    double logMin = log10(minVal);
    double logMax = log10(maxVal);
    double step = (num > 1) ? (logMax - logMin) / (num - 1) : 0.0;
    for (unsigned i = 0; i < num; i++) vec[i] = (float)pow(10.0, logMin + i * step);
    return vec;
}

} // namespace sp

#endif // KERNEL_CPU_H
