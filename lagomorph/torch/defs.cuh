#pragma once

const unsigned int MAX_THREADS_PER_BLOCK = 512;

// used in both interp and diff code
enum BackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
                          BACKGROUND_STRATEGY_ID,
                          BACKGROUND_STRATEGY_PARTIAL_ZERO,
                          BACKGROUND_STRATEGY_ZERO,
                          BACKGROUND_STRATEGY_CLAMP,
                          BACKGROUND_STRATEGY_WRAP,
                          BACKGROUND_STRATEGY_VAL};

const auto DEFAULT_BACKGROUND_STRATEGY = BACKGROUND_STRATEGY_CLAMP;
