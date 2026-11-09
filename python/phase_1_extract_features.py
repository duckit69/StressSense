"""phase_1_extract_features.py - Phase 1, step 2: compute features + quality metrics for every window.
Reads  data/windows/w<W>_h<H>/Sx_windows.npz
Writes data/features/w<W>_h<H>.csv   (subject, t_start, label, all features, all q_ metrics)
Log    logs/SSHRC_D1_features_w<W>_h<H>.csv  (per subject: windows, NaN counts, medians)
Usage (from the repository root):  python python/phase_1_extract_features.py
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import csv

import numpy as np
import pandas as pd

import config as C
from runlog import RunLog
from features import window_features


# Computes the features of every saved window of every subject (one row per window).
def extract(tag):
    rows = []
    for sid in C.SUBJECTS:
        a = np.load(C.WIN_DIR / tag / f"{sid}_windows.npz")
        for i in range(len(a["label"])):
            f = window_features(a["bvp"][i], a["eda"][i], a["temp"][i], a["acc"][i])
            rows.append({"subject": sid, "t_start": float(a["t_start"][i]),
                         "label": int(a["label"][i]), **f})
    return pd.DataFrame(rows)


def main():
    # The selected setting only: 30 s window, 10 s hop.
    configs = [(C.WINDOW_S, C.HOP_S)]
    C.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)

    for w, h in configs:
        tag = f"w{w}_h{h}"
        # One feature table per window setting, plus a per-subject summary log.
        df = extract(tag)
        df.to_csv(C.FEAT_DIR / f"{tag}.csv", index=False)
        feat_cols = [c for c in df.columns if c not in ("subject", "t_start", "label")]
        log = C.LOG_DIR / f"SSHRC_D1_features_{tag}.csv"
        with open(log, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["subject", "n_windows", "n_windows_any_nan", "q_bvp_valid_frac_median",
                         "bvp_hr_median", "eda_mean_median", "q_acc_sat_frac_mean"])
            for sid, g in df.groupby("subject", sort=False):
                wr.writerow([sid, len(g), int(g[feat_cols].isna().any(axis=1).sum()),
                             f"{g['q_bvp_valid_frac'].median():.3f}", f"{g['bvp_hr'].median():.1f}",
                             f"{g['eda_mean'].median():.3f}", f"{g['q_acc_sat_frac'].mean():.5f}"])
        nan_cols = df[feat_cols].isna().sum()
        nan_txt = ", ".join(f"{c}={n}" for c, n in nan_cols.items() if n) or "none"
        print(f"{tag:8s} windows={len(df):5d} features={len(feat_cols)}  NaN: {nan_txt}")
        print(f"         median valid-IBI fraction={df['q_bvp_valid_frac'].median():.3f}  "
              f"median HR={df['bvp_hr'].median():.1f} bpm  -> {log.name}")


if __name__ == "__main__":
    with RunLog("phase_1_extract_features"):
        main()
