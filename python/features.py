"""features.py - Feature (F4) and quality-metric (F3) functions for one window.
Pure functions, no I/O. Algorithms kept simple so the firmware C port can match them.
Names: <modality>_<feature> = classifier candidates; q_<...> = quality metrics only.
Used by phase_1_extract_features.py (and later by the replay agreement test, M5).
"""
import numpy as np
from scipy.signal import butter, sosfilt

import config as C

# Band-pass filter for the pulse signal, designed once (second-order sections, as on the ESP32).
_SOS_BVP = butter(C.BVP_FILT_ORDER, C.BVP_BAND_HZ, btype="bandpass",
                  fs=C.WRIST_FS["BVP"], output="sos")


def _slope(x, fs):
    """Least-squares slope in units per second."""
    t = np.arange(len(x)) / fs
    t = t - t.mean()
    return float(np.dot(t, x - x.mean()) / np.dot(t, t))


def longest_flat_s(x, fs):
    """Duration (s) of the longest run of identical consecutive samples (stuck sensor)."""
    same = np.diff(np.asarray(x)) == 0
    if not same.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], same.astype(int), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float((runs.max() + 1) / fs)


def detect_beats(bvp):
    """Band-pass (causal), then local maxima above K*std with a refractory period.
    Returns peak indices into the filtered signal (after the skipped transient)."""
    fs = C.WRIST_FS["BVP"]
    # Filter the pulse, then drop the first BVP_SKIP_S seconds where the filter is still settling.
    y = sosfilt(_SOS_BVP, bvp.astype(np.float64))[int(C.BVP_SKIP_S * fs):]
    # Candidate beats: local maxima higher than K times the standard deviation of the filtered signal.
    thr = C.BVP_PEAK_K * y.std()
    c = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thr)) + 1
    refr = int(C.IBI_MIN_S * fs)
    peaks = []
    for i in c:                                   # keep the higher peak inside the refractory period
        if peaks and i - peaks[-1] < refr:
            if y[i] > y[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    return np.asarray(peaks, dtype=int)


# Heart-rate features from the pulse (BVP) and the BVP quality metrics.
def bvp_features(bvp):
    fs = C.WRIST_FS["BVP"]
    p = detect_beats(bvp)
    # Inter-beat intervals in seconds.
    ibi = np.diff(p) / fs
    n_raw = len(ibi)
    # Keep plausible intervals, then drop those far from the window median (missed or extra beats).
    ok = (ibi >= C.IBI_MIN_S) & (ibi <= C.IBI_MAX_S)
    if ok.any():
        med = np.median(ibi[ok])
        ok &= np.abs(ibi - med) <= C.IBI_MAX_DEV * med
    v = ibi[ok]
    f = {"q_bvp_n_beats": len(p),
         "q_bvp_valid_frac": float(ok.mean()) if n_raw else 0.0,
         "q_bvp_ibi_cv": float(ibi.std() / ibi.mean()) if n_raw >= 2 else np.nan,   # all raw IBIs
         "q_bvp_flat_s": longest_flat_s(bvp, fs)}
    # Heart rate, SDNN and RMSSD need at least 3 valid intervals; otherwise they are missing (NaN).
    if len(v) >= 3:
        f["bvp_hr"] = 60.0 / v.mean()
        f["bvp_sdnn"] = 1000.0 * v.std()
        # RMSSD only over successive intervals that are both valid
        succ = ok[1:] & ok[:-1]
        d = np.diff(ibi)[succ]
        f["bvp_rmssd"] = 1000.0 * np.sqrt(np.mean(d ** 2)) if len(d) else np.nan
    else:
        f["bvp_hr"] = f["bvp_sdnn"] = f["bvp_rmssd"] = np.nan
    return f


# Skin-conductance (EDA) features: level, variability, trend and phasic responses (SCR).
def eda_features(eda):
    fs = C.WRIST_FS["EDA"]
    x = eda.astype(np.float64)
    k = C.EDA_SMOOTH_N
    # 1-second moving average to remove sample-to-sample noise before peak detection.
    s = np.convolve(x, np.ones(k) / k, mode="valid")
    # phasic responses: rise >= SCR_MIN_AMP_US from the last local minimum within SCR_MAX_RISE_S
    n_scr, amp_sum = 0, 0.0
    max_rise = int(C.SCR_MAX_RISE_S * fs)
    i_min = 0
    for i in range(1, len(s) - 1):
        if s[i] <= s[i - 1] and s[i] < s[i + 1]:
            i_min = i
        elif s[i] > s[i - 1] and s[i] >= s[i + 1]:
            rise = s[i] - s[i_min]
            if rise >= C.SCR_MIN_AMP_US and i - i_min <= max_rise:
                n_scr += 1
                amp_sum += rise
    # SCR rate is expressed per minute.
    minutes = len(x) / fs / 60.0
    return {"eda_mean": x.mean(), "eda_std": x.std(), "eda_slope": _slope(x, fs),
            "eda_scr_rate": n_scr / minutes, "eda_scr_amp_sum": amp_sum,
            "q_eda_std": x.std(), "q_eda_min": x.min(), "q_eda_max": x.max(),
            "q_eda_flat_s": longest_flat_s(x, fs)}


# Skin-temperature features (level and trend) plus its range for the quality check.
def temp_features(temp):
    fs = C.WRIST_FS["TEMP"]
    x = temp.astype(np.float64)
    return {"temp_mean": x.mean(), "temp_std": x.std(), "temp_slope": _slope(x, fs),
            "q_temp_min": x.min(), "q_temp_max": x.max()}


# Accelerometer features. acc_activity (mean change of the acceleration magnitude, g/s)
# is the effort-gate input; it is NOT given to the stress classifier.
def acc_features(acc):
    fs = C.WRIST_FS["ACC"]
    raw = acc.astype(np.float64)
    mag = np.sqrt(np.sum((raw * C.ACC_UNIT_G) ** 2, axis=1))      # in g
    return {"acc_mag_mean": mag.mean(), "acc_mag_std": mag.std(),
            "acc_activity": np.mean(np.abs(np.diff(mag))) * fs,     # g/s, effort-gate input (F6)
            "q_acc_sat_frac": float(np.mean((raw <= -128) | (raw >= 127)))}


# All features and quality metrics of one 30 s window, as one dictionary.
def window_features(bvp, eda, temp, acc):
    f = {}
    f.update(bvp_features(bvp))
    f.update(eda_features(eda))
    f.update(temp_features(temp))
    f.update(acc_features(acc))
    return f
