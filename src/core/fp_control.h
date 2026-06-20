#pragma once

#include <cstdint>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#include <float.h>
#else
#include <fenv.h>
#endif

namespace synchrotron {

struct FPControlState {
    uint32_t mxcsr;
    bool valid;
};

inline bool is_subnormal(double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint64_t exponent = (bits >> 52) & 0x7FF;
    uint64_t mantissa = bits & 0x000FFFFFFFFFFFFFULL;
    return (exponent == 0) && (mantissa != 0);
}

inline double flush_subnormal(double val) {
    if (is_subnormal(val)) return 0.0;
    return val;
}

inline double safe_exp(double x) {
    double result = std::exp(x);
    return flush_subnormal(result);
}

inline double safe_mul(double a, double b) {
    double result = a * b;
    return flush_subnormal(result);
}

inline FPControlState enable_daz_ftz() {
    FPControlState state;
    state.valid = false;

#ifdef _MSC_VER
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    state.mxcsr = _mm_getcsr();
    state.valid = true;
#else
    if (fesetround(FE_TONEAREST) == 0) {
#ifdef __SSE__
        unsigned int mxcsr;
        __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
        state.mxcsr = mxcsr;
        mxcsr |= (1 << 15);
        mxcsr |= (1 << 6);
        __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
        state.valid = true;
#endif
    }
#endif
    return state;
}

inline void restore_fp_state(const FPControlState& state) {
    if (!state.valid) return;

#ifdef _MSC_VER
    _mm_setcsr(state.mxcsr);
#else
#ifdef __SSE__
    __asm__ __volatile__("ldmxcsr %0" : : "m"(state.mxcsr));
#endif
#endif
}

class ScopedFTZ {
public:
    ScopedFTZ() : saved_(enable_daz_ftz()) {}
    ~ScopedFTZ() { restore_fp_state(saved_); }
    ScopedFTZ(const ScopedFTZ&) = delete;
    ScopedFTZ& operator=(const ScopedFTZ&) = delete;
private:
    FPControlState saved_;
};

}
