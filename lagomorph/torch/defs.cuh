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
