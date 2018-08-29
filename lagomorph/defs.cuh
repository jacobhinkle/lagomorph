#ifndef DEFS_CUH
#define DEFS_CUH

const unsigned int MAX_THREADS_PER_BLOCK = 1024;

// used in both interp and diff code
enum BackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
                          BACKGROUND_STRATEGY_ID,
                          BACKGROUND_STRATEGY_PARTIAL_ZERO,
                          BACKGROUND_STRATEGY_ZERO,
                          BACKGROUND_STRATEGY_CLAMP,
                          BACKGROUND_STRATEGY_WRAP,
                          BACKGROUND_STRATEGY_VAL};

#endif /* DEFS_CUH */
