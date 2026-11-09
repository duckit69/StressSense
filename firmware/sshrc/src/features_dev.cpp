// features_dev.cpp - see features_dev.h. Mirrors python/features.py function by function.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "features_dev.h"
#include "bvp_filter.h"
#ifdef ARDUINO
#include <Arduino.h>
#define NOW_US() ((int32_t)esp_timer_get_time())
#else
#include <chrono>
static int32_t NOW_US() { return (int32_t)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
#endif

// Feature order sent to the host and used by decision.cpp (same as config.DEVICE_FEATURES).
const char *DEV_FEAT_NAMES[N_DEV_FEAT] = {
  "q_bvp_n_beats", "q_bvp_valid_frac", "q_bvp_ibi_cv", "q_bvp_flat_s", "bvp_hr", "bvp_sdnn", "bvp_rmssd",
  "eda_mean", "eda_std", "eda_slope", "eda_scr_rate", "eda_scr_amp_sum", "q_eda_std", "q_eda_min", "q_eda_max",
  "q_eda_flat_s", "temp_mean", "temp_std", "temp_slope", "q_temp_min", "q_temp_max",
  "acc_mag_mean", "acc_mag_std", "acc_activity", "q_acc_sat_frac"};

// Static work buffers (no dynamic memory on the device).
static float s_y[WIN_N_BVP];             // filtered BVP
static int16_t s_peaks[WIN_N_BVP / 2];
static float s_ibi[WIN_N_BVP / 2];
static uint8_t s_ok[WIN_N_BVP / 2];
static float s_tmp[WIN_N_BVP / 2];
static float s_mag[WIN_N_ACC];
static double s_sm[WIN_N_EDA];          // double: avoids float32 ties in SCR detection (phase 3.3)

// Small statistics helpers, written to give the same numbers as numpy.
static float mean_f(const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return (float)(s / n); }
static float std_f(const float *x, int n) {          // population SD, two-pass (numpy default)
  float m = mean_f(x, n); double s = 0;
  for (int i = 0; i < n; i++) { double d = x[i] - m; s += d * d; }
  return (float)sqrt(s / n);
}
static float slope_f(const float *x, int n, float fs) {   // least squares, units per second
  float xm = mean_f(x, n); double tm = (n - 1) / (2.0 * fs), num = 0, den = 0;
  for (int i = 0; i < n; i++) { double t = i / (double)fs - tm; num += t * (x[i] - xm); den += t * t; }
  return (float)(num / den);
}
static float longest_flat_s(const float *x, int n, float fs) {
  int best = 0, run = 0;
  for (int i = 1; i < n; i++) { if (x[i] == x[i - 1]) { run++; if (run > best) best = run; } else run = 0; }
  return best ? (best + 1) / fs : 0.0f;
}
static int cmp_f(const void *a, const void *b) { float d = *(const float *)a - *(const float *)b; return (d > 0) - (d < 0); }
static float median_f(float *x, int n) {             // sorts x in place; numpy median (mean of middle two)
  qsort(x, n, sizeof(float), cmp_f);
  return (n % 2) ? x[n / 2] : 0.5f * (x[n / 2 - 1] + x[n / 2]);
}

// Band-pass filter the pulse (same coefficients as Python, bvp_filter.h) and find beats:
// local maxima above K x SD, keeping the higher peak inside the refractory period.
static int detect_beats(const float *bvp, StageTimes *t) {
  int32_t t0 = NOW_US();
  float z[N_SOS][2] = {{0}};
  for (int i = 0; i < WIN_N_BVP; i++) {               // sosfilt, DF-II transposed, zero initial state
    float v = bvp[i];
    for (int s = 0; s < N_SOS; s++) {
      const float *c = BVP_SOS[s];
      float y = c[0] * v + z[s][0];
      z[s][0] = c[1] * v - c[3] * y + z[s][1];
      z[s][1] = c[2] * v - c[4] * y;
      v = y;
    }
    s_y[i] = v;
  }
  t->bvp_filter_us = NOW_US() - t0;
  const int skip = (int)(BVP_SKIP_S * FS_BVP);
  const float *y = s_y + skip;
  const int n = WIN_N_BVP - skip;
  float thr = BVP_PEAK_K * std_f(y, n);
  const int refr = (int)(IBI_MIN_S * FS_BVP);
  int np = 0;
  for (int i = 1; i < n - 1; i++) {
    if (y[i] > y[i - 1] && y[i] >= y[i + 1] && y[i] > thr) {
      if (np && i - s_peaks[np - 1] < refr) { if (y[i] > y[s_peaks[np - 1]]) s_peaks[np - 1] = (int16_t)i; }
      else s_peaks[np++] = (int16_t)i;
    }
  }
  return np;
}

// Outputs [0..6]: beat count, valid-interval fraction, interval CV, flat time, HR, SDNN, RMSSD.
static void bvp_features(const float *bvp, float *o, StageTimes *t) {
  int np = detect_beats(bvp, t);
  int32_t t0 = NOW_US();
  int nr = np > 0 ? np - 1 : 0;
  int nok = 0;
  for (int j = 0; j < nr; j++) {
    s_ibi[j] = (s_peaks[j + 1] - s_peaks[j]) / (float)FS_BVP;
    s_ok[j] = s_ibi[j] >= IBI_MIN_S && s_ibi[j] <= IBI_MAX_S;
    if (s_ok[j]) s_tmp[nok++] = s_ibi[j];
  }
  if (nok) {
    float med = median_f(s_tmp, nok);
    for (int j = 0; j < nr; j++) if (s_ok[j] && fabsf(s_ibi[j] - med) > IBI_MAX_DEV * med) s_ok[j] = 0;
  }
  int nv = 0; for (int j = 0; j < nr; j++) if (s_ok[j]) s_tmp[nv++] = s_ibi[j];
  o[0] = (float)np;
  o[1] = nr ? (float)nv / nr : 0.0f;
  o[2] = nr >= 2 ? std_f(s_ibi, nr) / mean_f(s_ibi, nr) : NAN;
  o[3] = longest_flat_s(bvp, WIN_N_BVP, FS_BVP);
  if (nv >= 3) {
    o[4] = 60.0f / mean_f(s_tmp, nv);
    o[5] = 1000.0f * std_f(s_tmp, nv);
    double s = 0; int nd = 0;
    for (int j = 0; j + 1 < nr; j++) if (s_ok[j] && s_ok[j + 1]) { double d = s_ibi[j + 1] - s_ibi[j]; s += d * d; nd++; }
    o[6] = nd ? (float)(1000.0 * sqrt(s / nd)) : NAN;
  } else { o[4] = o[5] = o[6] = NAN; }
  t->bvp_beats_us = NOW_US() - t0 + 0;
}

// Outputs [7..15]: EDA mean, SD, slope, SCR rate and amplitude, and quality metrics.
static void eda_features(const float *x, float *o) {
  const int n = WIN_N_EDA, k = EDA_SMOOTH_N, m = n - k + 1;
  for (int j = 0; j < m; j++) { double s = 0; for (int i = 0; i < k; i++) s += (double)x[j + i] * (1.0 / k); s_sm[j] = s; }
  int n_scr = 0, i_min = 0; double amp = 0;
  const int max_rise = (int)(SCR_MAX_RISE_S * FS_EDA);
  for (int i = 1; i < m - 1; i++) {
    if (s_sm[i] <= s_sm[i - 1] && s_sm[i] < s_sm[i + 1]) i_min = i;
    else if (s_sm[i] > s_sm[i - 1] && s_sm[i] >= s_sm[i + 1]) {
      double rise = s_sm[i] - s_sm[i_min];
      if (rise >= SCR_MIN_AMP_US && i - i_min <= max_rise) { n_scr++; amp += rise; }
    }
  }
  float mn = x[0], mx = x[0]; for (int i = 1; i < n; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; }
  float sd = std_f(x, n);
  o[0] = mean_f(x, n); o[1] = sd; o[2] = slope_f(x, n, FS_EDA);
  o[3] = n_scr / (n / (float)FS_EDA / 60.0f); o[4] = (float)amp;
  o[5] = sd; o[6] = mn; o[7] = mx; o[8] = longest_flat_s(x, n, FS_EDA);
}

// Outputs [16..20]: temperature mean, SD, slope, min, max.
static void temp_features(const float *x, float *o) {
  const int n = WIN_N_TEMP;
  float mn = x[0], mx = x[0]; for (int i = 1; i < n; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; }
  o[0] = mean_f(x, n); o[1] = std_f(x, n); o[2] = slope_f(x, n, FS_TEMP); o[3] = mn; o[4] = mx;
}

// Outputs [21..24]: acceleration magnitude mean and SD, activity index (g/s), saturated fraction.
static void acc_features(const int8_t *a, float *o) {
  const int n = WIN_N_ACC; int sat = 0;
  for (int i = 0; i < n; i++) {
    float gx = a[3 * i] * ACC_UNIT_G, gy = a[3 * i + 1] * ACC_UNIT_G, gz = a[3 * i + 2] * ACC_UNIT_G;
    s_mag[i] = sqrtf(gx * gx + gy * gy + gz * gz);
    for (int c = 0; c < 3; c++) if (a[3 * i + c] <= -128 || a[3 * i + c] >= 127) sat++;
  }
  double d = 0; for (int i = 1; i < n; i++) d += fabsf(s_mag[i] - s_mag[i - 1]);
  o[0] = mean_f(s_mag, n); o[1] = std_f(s_mag, n); o[2] = (float)(d / (n - 1) * FS_ACC); o[3] = (float)sat / (3 * n);
}

// All 25 features of one window, with the time spent in each stage.
void dev_features(const float *bvp, const float *eda, const float *temp, const int8_t *acc,
                  float out[N_DEV_FEAT], StageTimes *t) {
  int32_t t0 = NOW_US();
  bvp_features(bvp, out + 0, t);
  int32_t t1 = NOW_US(); eda_features(eda, out + 7); t->eda_us = NOW_US() - t1;
  t1 = NOW_US(); temp_features(temp, out + 16); t->temp_us = NOW_US() - t1;
  t1 = NOW_US(); acc_features(acc, out + 21); t->acc_us = NOW_US() - t1;
  t->total_us = NOW_US() - t0;
}
