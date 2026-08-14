#ifndef SUPPORT_H
#define SUPPORT_H

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

static void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

static double elapsedTime(Timer timer) {
    return ((double)((timer.endTime.tv_sec - timer.startTime.tv_sec)
        + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

void initParamVector(double** vec_h, unsigned num, double minVal, double maxVal);
void verify(double* L_h, double* C_h, double* R_h, double* fsw_h, unsigned numL, unsigned numC, unsigned numR, unsigned numFsw, unsigned numHarmonics, double* worstAtten_h, double* thd_h, unsigned totalCombinations);

#endif // SUPPORT_H
