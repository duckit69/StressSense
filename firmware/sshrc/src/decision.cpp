// decision.cpp - see decision.h.
#include <math.h>
#include "decision.h"
#include "classifier.h"
#include "config.h"

// Session state, kept in RAM only (lost at power-off by design):
// running sums for the calibration means, current state, pending state change,
// confidence held during effort, and the frame sequence number.
static double cal_sum[N_FEAT];
static int cal_n[N_FEAT];
static float cal_mean[N_FEAT];
static int n_windows = 0;
static int st_state = 0, pend_t = -1, pend_n = 0;
static float held = 1.0f;
static uint16_t seq = 0;

// CRC-16/CCITT-FALSE over the first 6 frame bytes (lets the robot side reject corrupted frames).
uint16_t crc16_ccitt(const uint8_t *d, int n) {
  uint16_t c = 0xFFFF;
  for (int i = 0; i < n; i++) {
    c ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
  }
  return c;
}

// New session: forget the calibration and the state (the next 28 windows recalibrate).
void dec_new_session() {
  for (int i = 0; i < N_FEAT; i++) { cal_sum[i] = 0; cal_n[i] = 0; cal_mean[i] = 0; }
  n_windows = 0; st_state = 0; pend_t = -1; pend_n = 0; held = 1.0f; seq = 0;
}

// 0 = good, 1 = ACC degraded, 2 = a classifier channel unusable. Indexes refer to DEV_FEAT_NAMES.
static uint8_t quality_flag(const float *f) {     // python/quality.py (IBI-CV rule disabled)
  bool bvp = f[0] < Q_BVP_MIN_BEATS || f[1] < Q_BVP_MIN_VALID_FRAC || f[3] >= Q_BVP_MAX_FLAT_S;
  bool eda = f[12] < Q_EDA_FLAT_STD_US || f[13] < Q_EDA_MIN_US || f[15] >= Q_EDA_MAX_FLAT_S;
  bool temp = f[19] < Q_TEMP_MIN_C || f[20] > Q_TEMP_MAX_C;
  bool acc = f[24] > Q_ACC_MAX_SAT_FRAC;
  return (bvp || eda || temp) ? 2 : (acc ? 1 : 0);
}

// State the probability p points to, given the current state s. Entry thresholds
// (0.5, 0.8) are higher than exit thresholds (0.4, 0.7): hysteresis.
static int target_of(int s, float p) {             // python/state.py:_target
  if (s == 0) return p >= P_HIGH_ON ? 2 : (p >= P_ELEV_ON ? 1 : 0);
  if (s == 1) return p >= P_HIGH_ON ? 2 : (p < P_ELEV_OFF ? 0 : 1);
  return p >= P_HIGH_OFF ? 2 : (p >= P_ELEV_OFF ? 1 : 0);
}

// Writes the 8-byte frame: flags, confidence, sequence number, reason, reserved, CRC.
static void pack(DecOut *o) {
  o->conf_byte = (uint8_t)lrintf(fminf(fmaxf(o->conf, 0.0f), 1.0f) * 255.0f);
  o->frame[0] = (uint8_t)((1 << 6) | (o->state << 4) | (o->qflag << 2) | (o->effort << 1) | o->fallback);
  o->frame[1] = o->conf_byte;
  o->frame[2] = (uint8_t)(o->seq & 0xFF);
  o->frame[3] = (uint8_t)(o->seq >> 8);
  o->frame[4] = o->reason;
  o->frame[5] = 0;
  uint16_t c = crc16_ccitt(o->frame, 6);
  o->frame[6] = (uint8_t)(c & 0xFF);
  o->frame[7] = (uint8_t)(c >> 8);
}

// One decision per window. Order: calibration -> classifier -> bad quality -> effort -> state machine.
void dec_step(const float f[N_DEV_FEAT], DecOut *o) {
  o->qflag = quality_flag(f);
  // Effort gate input: acc_activity (feature 23) above the exported threshold.
  o->effort = f[23] > GATE_ACT_THR_GPS;
  o->seq = seq++;
  o->acc = 0; o->p = NAN;
  if (n_windows < N_CALIB_WINDOWS) {                // calibration: accumulate, report "calibrating"
    for (int i = 0; i < N_FEAT; i++) {
      float v = f[FEAT_DEV_INDEX[i]];
      if (!isnan(v)) { cal_sum[i] += v; cal_n[i]++; }
    }
    if (++n_windows == N_CALIB_WINDOWS)
      for (int i = 0; i < N_FEAT; i++) cal_mean[i] = cal_n[i] ? (float)(cal_sum[i] / cal_n[i]) : NAN;
    o->state = 0; o->conf = 0; o->fallback = 1; o->reason = R_CALIBRATING;
    pack(o);
    return;
  }
  float x[N_FEAT];
  // Classifier input: the 11 classifier features minus this operator's calibration means.
  for (int i = 0; i < N_FEAT; i++) x[i] = f[FEAT_DEV_INDEX[i]] - cal_mean[i];   // NaN stays NaN -> median
  float logit;
  o->acc = sshrc_classify(x, &logit);
  o->p = 1.0f / (1.0f + expf(-logit));
  // Bad signal: keep the last state, confidence 0, ask the robot side to fall back.
  if (o->qflag == 2) {
    o->state = st_state; o->conf = 0; o->fallback = 1; o->reason = R_QUALITY;
  // Physical effort: do not update the state; lower the confidence each window; fall back below CONF_MIN.
  } else if (o->effort) {
    held *= GATE_DECAY;
    o->state = st_state; o->conf = held; o->fallback = held < CONF_MIN;
    o->reason = o->fallback ? R_EFFORT_LOW_CONF : R_EFFORT_HOLD;
    pend_t = -1; pend_n = 0;
  } else {
    // Normal window: change state only after N_DWELL consecutive windows point to the same new state.
    int t = target_of(st_state, o->p);
    if (t != st_state) {
      pend_n = (t == pend_t) ? pend_n + 1 : 1;
      pend_t = t;
      if (pend_n >= N_DWELL) { st_state = t; pend_t = -1; pend_n = 0; }
    } else { pend_t = -1; pend_n = 0; }
    held = 2.0f * fabsf(o->p - 0.5f);
    o->state = st_state; o->conf = held; o->fallback = 0; o->reason = R_NORMAL;
  }
  pack(o);
}
