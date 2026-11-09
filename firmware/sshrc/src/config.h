// config.h - StressSense-HRC parameters (firmware side).
// Mirror of python/config.py: change both together. (Moved from firmware/include for the Arduino IDE.)
// [WESAD readme] = data/WESAD/wesad_readme.pdf ; [T] = design target.
#pragma once

// Sampling rates in Hz [WESAD readme]
#define FS_ACC   32
#define FS_BVP   64
#define FS_EDA   4
#define FS_TEMP  4
#define ACC_UNIT_G (1.0f / 64.0f)   // wrist ACC raw unit = 1/64 g

// Windowing: FROZEN after the phase-1 sweep (run 20261101-154125_d1_window_sweep)
#define WINDOW_S 30
#define HOP_S    10

// Samples per window (derived: WINDOW_S * FS)
#define WIN_N_ACC  (WINDOW_S * FS_ACC)    // 960
#define WIN_N_BVP  (WINDOW_S * FS_BVP)    // 1920
#define WIN_N_EDA  (WINDOW_S * FS_EDA)    // 120
#define WIN_N_TEMP (WINDOW_S * FS_TEMP)   // 120

// Feature extraction (F4) - mirror of python/config.py
#define BVP_BAND_LO_HZ  0.7f
#define BVP_BAND_HI_HZ  3.5f
#define BVP_FILT_ORDER  2
#define BVP_SKIP_S      2.0f
#define BVP_PEAK_K      0.5f
#define IBI_MIN_S       0.30f
#define IBI_MAX_S       1.50f
#define IBI_MAX_DEV     0.30f
#define EDA_SMOOTH_N    4
#define SCR_MIN_AMP_US  0.01   // double on purpose: compared in double, like Python
#define SCR_MAX_RISE_S  4.0f

// Signal quality (F3) - mirror of python/config.py
#define Q_HR_MIN_BPM          40
#define Q_BVP_MIN_VALID_FRAC  0.5f
// Q_BVP_MAX_IBI_CV disabled (see python/config.py)
#define Q_BVP_MAX_FLAT_S      1.0f
#define Q_EDA_MIN_US          0.01f
#define Q_EDA_FLAT_STD_US     1e-6f
#define Q_EDA_MAX_FLAT_S      10.0f
#define Q_TEMP_MIN_C          20.0f
#define Q_TEMP_MAX_C          42.0f
#define Q_ACC_MAX_SAT_FRAC    0.05f

// Effort gate (F6) and state mapping (F7)
#define GATE_DECAY   0.8f
#define CONF_MIN     0.5f
#define P_ELEV_ON    0.5f
#define P_ELEV_OFF   0.4f
#define P_HIGH_ON    0.8f
#define P_HIGH_OFF   0.7f
#define N_DWELL      2
// gate threshold (g/s) is exported with the trained model on phase 2.4

// Per-operator baseline normalisation (F7b)
#define NORM_CALIB_S 300            // first 5 min of calm calibration
#define N_CALIB_WINDOWS ((NORM_CALIB_S - WINDOW_S) / HOP_S + 1)   // windows fully inside it (28 at 30/10)
#define Q_BVP_MIN_BEATS (WINDOW_S * Q_HR_MIN_BPM / 60)              // 20 at 30 s

// Serial link to the laptop
#define SERIAL_BAUD 921600   // fallback 115200 (change python/config.py too)

// Phase 4: operational mode
#define LED_PIN 2                   // DevKit onboard LED (assumption): on = high stress or fallback

// Reproducibility
#define SEED 42

// Thresholds (quality, effort gate, state mapping) are added on phase 2.
