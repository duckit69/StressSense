// classifier.cpp - int8 inference, same arithmetic as python/quant.py:int8_logit.
// NaN -> training median; z = (x - mean) / scale (float32); q = rint(z / S_X) clipped to [-127, 127]
// (rint = round half to even, like numpy.round); acc = sum(q * QW) + QB in int32.
#include <math.h>
#include "classifier.h"

// Returns the int32 accumulator; class = stress if > 0. Optionally returns the logit.
int32_t sshrc_classify(const float x[N_FEAT], float *logit) {
  int32_t acc = QB;
  for (int i = 0; i < N_FEAT; i++) {
    float v = isnan(x[i]) ? FEAT_MEDIAN[i] : x[i];
    float z = (v - FEAT_MEAN[i]) / FEAT_SCALE[i];
    float qf = rintf(z / S_X);
    if (qf > 127.0f) qf = 127.0f;
    if (qf < -127.0f) qf = -127.0f;
    acc += (int32_t)qf * (int32_t)QW[i];
  }
  if (logit) *logit = (float)acc * S_X * S_W;
  return acc;
}
