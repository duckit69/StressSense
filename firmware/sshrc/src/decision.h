// decision.h - On-device decision chain after feature extraction (one call per 30 s window):
//   quality flag (F3) -> calibration means (F7b, first N_CALIB_WINDOWS windows of a session, RAM only)
//   -> int8 classifier -> effort gate (F6) -> 3-state machine with hysteresis/dwell (F7) -> 8-byte frame (F8).
// Mirrors python/quality.py, python/state.py and python/frame.py.
#pragma once
#include <stdint.h>
#include "features_dev.h"

enum Reason : uint8_t { R_NORMAL = 0, R_EFFORT_HOLD = 1, R_EFFORT_LOW_CONF = 2, R_QUALITY = 3, R_CALIBRATING = 4 };

struct DecOut {
  uint8_t qflag, effort, state, fallback, reason, conf_byte;
  uint16_t seq;
  int32_t acc;      // classifier accumulator (0 while calibrating)
  float p, conf;
  uint8_t frame[8]; // the ONLY data meant to leave the device
};

void dec_new_session();                                   // forget calibration, state and sequence number
void dec_step(const float f[N_DEV_FEAT], DecOut *o);
uint16_t crc16_ccitt(const uint8_t *d, int n);           // CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)
