#ifndef SUPPORT_CPU_H
#define SUPPORT_CPU_H

#include <cstdio>
#include <cstdlib>
#include <chrono>

// wall clock timer mirroring GPU version
struct Timer {
    std::chrono::high_resolution_clock::time_point start, end;
};

inline void startTime(Timer* t) { t->start = std::chrono::high_resolution_clock::now(); }
inline void stopTime(Timer* t)  { t->end   = std::chrono::high_resolution_clock::now(); }
inline double elapsedTime(const Timer& t) {
    return std::chrono::duration<double>(t.end - t.start).count();
}

#define FATAL(msg) \
    do { fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg); exit(-1); } while(0)

#endif // SUPPORT_CPU_H
