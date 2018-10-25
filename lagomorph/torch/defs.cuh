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

// Set lagomorph_debug to 1 to enable cuda error checking (causes synchronization)
#define lagomorph_debug 0

#define LAGOMORPH_CUDA_CHECK(file, line) \
  if (lagomorph_debug) { \
	cudaDeviceSynchronize(); \
	cudaError_t error = cudaGetLastError(); \
	if(error != cudaSuccess) \
		printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error)); \
  }

