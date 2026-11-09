import numpy as np
import pandas as pd

import config as C


# Feature columns that belong to the given modalities (by name prefix, e.g. 'eda_').
def feature_cols(df, modalities):
    prefixes = tuple(C.MODALITY_PREFIX[m] for m in modalities)
    return [c for c in df.columns if c.startswith(prefixes)]


def calibration_mask(df):
    """Boolean mask of calibration windows (first NORM_CALIB_S s of each subject's baseline)."""
    W = C.WINDOW_S
    cal = np.zeros(len(df), bool)
    for s in C.SUBJECTS:
        m = df["subject"] == s
        b0 = df.loc[m & (df["label"] == 1), "t_start"].min()
        cal |= (m & (df["label"] == 1) & (df["t_start"] >= b0)
                & (df["t_start"] + W <= b0 + C.NORM_CALIB_S)).to_numpy()
    return cal


def calibration_means(df, cols):
    """{subject: Series of calibration means} (used to normalise new or modified windows)."""
    cal = calibration_mask(df)
    return {s: df.loc[cal & (df["subject"] == s).to_numpy(), cols].mean() for s in C.SUBJECTS}


def task_table(task, normalise=C.FROZEN_NORM, tag=None):
    """Return (table with column y, all feature columns of all modalities)."""
    W = C.WINDOW_S
    tag = tag or f"w{C.WINDOW_S}_h{C.HOP_S}"
    df = pd.read_csv(C.FEAT_DIR / f"{tag}.csv")
    cols = feature_cols(df, C.MODALITY_PREFIX.keys())
    out = df.copy()
    is_cal = np.zeros(len(df), bool)
    for s in C.SUBJECTS:
        m = df["subject"] == s
        b0 = df.loc[m & (df["label"] == 1), "t_start"].min()
        cal = m & (df["label"] == 1) & (df["t_start"] >= b0) & (df["t_start"] + W <= b0 + C.NORM_CALIB_S)
        # Normalised variant: subtract this subject's calibration mean from all of its windows.
        if normalise:
            out.loc[m, cols] = df.loc[m, cols] - df.loc[cal, cols].mean()
        is_cal |= cal.to_numpy()
    # Calibration windows are never used for training or scoring.
    keep = (~is_cal) & df["label"].isin(task.keys()).to_numpy()
    out = out[keep].copy()
    out["y"] = out["label"].map(task).astype(int)
    return out.reset_index(drop=True), cols
