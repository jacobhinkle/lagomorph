/* vim: set ft=cuda: */
#pragma once

// used in both interp and diff code
enum BackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
                          BACKGROUND_STRATEGY_ID,
                          BACKGROUND_STRATEGY_PARTIAL_ZERO,
                          BACKGROUND_STRATEGY_ZERO,
                          BACKGROUND_STRATEGY_CLAMP,
                          BACKGROUND_STRATEGY_WRAP,
                          BACKGROUND_STRATEGY_VAL};

const auto DEFAULT_BACKGROUND_STRATEGY = BACKGROUND_STRATEGY_CLAMP;

extern bool lagomorph_debug_mode;

#define LAGOMORPH_CUDA_CHECK(file, line) \
  if (lagomorph_debug_mode) { \
	cudaDeviceSynchronize(); \
	cudaError_t error = cudaGetLastError(); \
	if(error != cudaSuccess) \
		printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error)); \
  }

#define LAGOMORPH_DISPATCH_BOOL(VALUE, PLACEHOLDERNAME, ...) \
	[&] { \
        if (VALUE) { \
            const bool PLACEHOLDERNAME = true; \
            return __VA_ARGS__(); \
        } else { \
            const bool PLACEHOLDERNAME = false; \
            return __VA_ARGS__(); \
        } \
    }()

// PyTorch 1.1 introduced TORCH_CHECK and deprecated AT_CHECK. The only known
// incompatibility with 1.0 is the use of TORCH_CHECK, and this macro
// definition should give us that amount of backward compatibility.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif


#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE 
#endif
