// classifier.h - int8 stress classifier (frozen logistic regression), parameters in model_params.h.
// Input: the N_FEAT features AFTER subtraction of the operator's calibration mean (NaN = missing).
#pragma once
#include <stdint.h>
#include "model_params.h"

// Returns the int32 accumulator; class = (acc > 0). *logit = acc * S_X * S_W (may be NULL).
int32_t sshrc_classify(const float x[N_FEAT], float *logit);
