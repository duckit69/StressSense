"""quality.py - Signal-quality flag (F3) from the q_ metrics of one or many windows.
flag 0 = good, 1 = degraded (ACC saturated: effort gate unreliable), 2 = bad (BVP, EDA or TEMP unusable).
Limits are fixed in config.py (physically motivated, not tuned on clean data).
"""
import numpy as np

import config as C


# One yes/no per channel: is it unusable in this window? Limits are fixed in config.py.
def channel_bad(q):
    """q: mapping or DataFrame with q_ columns. Returns dict of boolean arrays per channel."""
    g = lambda k: np.asarray(q[k], dtype=float)
    bvp = (g("q_bvp_n_beats") < C.Q_BVP_MIN_BEATS) | (g("q_bvp_valid_frac") < C.Q_BVP_MIN_VALID_FRAC) \
        | (g("q_bvp_flat_s") >= C.Q_BVP_MAX_FLAT_S)
    if C.Q_BVP_MAX_IBI_CV is not None:
        bvp |= np.nan_to_num(g("q_bvp_ibi_cv"), nan=np.inf) > C.Q_BVP_MAX_IBI_CV
    return {"bvp": bvp,
            "eda": (g("q_eda_std") < C.Q_EDA_FLAT_STD_US) | (g("q_eda_min") < C.Q_EDA_MIN_US)
                   | (g("q_eda_flat_s") >= C.Q_EDA_MAX_FLAT_S),
            "temp": (g("q_temp_min") < C.Q_TEMP_MIN_C) | (g("q_temp_max") > C.Q_TEMP_MAX_C),
            "acc": g("q_acc_sat_frac") > C.Q_ACC_MAX_SAT_FRAC}


# Combines the channels: any classifier channel bad -> 2, only ACC degraded -> 1, else 0.
def quality_flag(q):
    b = channel_bad(q)
    return np.where(b["bvp"] | b["eda"] | b["temp"], 2, np.where(b["acc"], 1, 0)).astype(int)
