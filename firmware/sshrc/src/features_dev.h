// features_dev.h - On-device feature + quality-metric extraction for one 30 s window.
// Line-by-line port of python/features.py (float32 here, float64 in Python).
// Output order = DEVICE_FEATURES in python/config.py (N_DEV_FEAT values).
#pragma once
#include <stdint.h>
#include "config.h"

#define N_DEV_FEAT 25
extern const char *DEV_FEAT_NAMES[N_DEV_FEAT];

struct StageTimes { int32_t bvp_filter_us, bvp_beats_us, eda_us, temp_us, acc_us, total_us; };

// bvp[WIN_N_BVP], eda[WIN_N_EDA], temp[WIN_N_TEMP]: float; acc[WIN_N_ACC * 3]: raw int8 (x,y,z interleaved)
void dev_features(const float *bvp, const float *eda, const float *temp, const int8_t *acc,
                  float out[N_DEV_FEAT], StageTimes *t);
