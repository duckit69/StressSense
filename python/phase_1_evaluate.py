"""phase_1_evaluate.py - Phase 1, step 3: evaluate the selected stress classifier on WESAD.
Model: logistic regression (class_weight="balanced"), 11 features from BVP + EDA + skin temperature,
30 s windows every 10 s, per-operator calibration (each feature minus the subject's mean over the first
5 min of baseline; calibration windows are never trained on or scored). Task: stress vs non-stress
(baseline + amusement). Test: leave-one-subject-out, each of the 15 subjects is the unseen test subject once.
Logs: logs/SSHRC_L2_calibration.csv  (subject: calibration windows, start, feature means)
      logs/SSHRC_L2_evaluation.csv   (subject: confusion counts, accuracy, macro-F1, stress onset latency)
Usage (from the repository root):  python python/phase_1_evaluate.py
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import csv

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from calib import calibration_mask, feature_cols, task_table
from loso import one_fold, onset_latency
from runlog import RunLog

W, H = C.WINDOW_S, C.HOP_S
TAG = f"w{W}_h{H}"


# Writes a list of dictionaries to a CSV log.
def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Data: calibrated features of all subjects, calibration windows removed.
    df, all_cols = task_table(C.TASK_BINARY, normalise=True)
    cols = feature_cols(df[all_cols], C.CLASSIFIER_MODALITIES)

    # Log the calibration of each subject (how many windows, which feature means were subtracted).
    raw = pd.read_csv(C.FEAT_DIR / f"{TAG}.csv")
    cal = calibration_mask(raw)
    calib_rows = []
    for s in C.SUBJECTS:
        m = cal & (raw["subject"] == s).to_numpy()
        mu = raw.loc[m, cols].mean()
        calib_rows.append({"subject": s, "baseline_first_t_start_s": raw.loc[m, "t_start"].min(),
                           "n_calib_windows": int(m.sum()), **{f"mean_{c}": mu[c] for c in cols}})
    write(C.LOG_DIR / "SSHRC_L2_calibration.csv", calib_rows)
    n_cal = [r["n_calib_windows"] for r in calib_rows]
    print(f"Model: logistic regression, window {TAG}, per-operator calibration")
    print(f"Features ({len(cols)}): {', '.join(cols)}")
    print(f"Calibration: first {C.NORM_CALIB_S} s of baseline, windows per subject min={min(n_cal)} max={max(n_cal)}")
    print(f"Scored windows: {len(df)} (stress={int(df['y'].sum())})\n")

    # 2. Leave-one-subject-out: train on 14 subjects, test on the 15th, for every subject (in parallel).
    out = Parallel(n_jobs=C.N_JOBS)(delayed(one_fold)(df, cols, s, [x for x in C.SUBJECTS if x != s])
                                    for s in C.SUBJECTS)
    rows = []
    print(f"{'subject':8s} {'windows':>7s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'TP':>5s} {'acc':>6s} {'macro-F1':>8s} {'onset s':>8s}")
    for s, (met, y_pred, te) in zip(C.SUBJECTS, out):
        lat = onset_latency(te, y_pred, W, H)
        rows.append({"subject": s, **met, "onset_latency_s": lat})
        print(f"{s:8s} {met['n']:7d} {met['tn']:5d} {met['fp']:5d} {met['fn']:5d} {met['tp']:5d} "
              f"{met['accuracy']:6.3f} {met['macro_f1']:8.3f} {lat:8.0f}")
    write(C.LOG_DIR / "SSHRC_L2_evaluation.csv", rows)

    # 3. Summary over the 15 test subjects.
    f1 = np.array([r["macro_f1"] for r in rows])
    tot = {q: sum(r[q] for r in rows) for q in ("tn", "fp", "fn", "tp")}
    recall = tot["tp"] / (tot["tp"] + tot["fn"])
    false_alarm = tot["fp"] / (tot["fp"] + tot["tn"])
    print(f"\nLOSO over {len(f1)} subjects: macro-F1 = {f1.mean():.3f} +/- {f1.std(ddof=1):.3f} (SD), "
          f"SE {f1.std(ddof=1) / np.sqrt(len(f1)):.3f}")
    print(f"Pooled: TN={tot['tn']} FP={tot['fp']} FN={tot['fn']} TP={tot['tp']} | "
          f"stress recall {100 * recall:.1f} %, false-alarm rate {100 * false_alarm:.1f} %")
    print(f"Stress onset latency: median {np.nanmedian([r['onset_latency_s'] for r in rows]):.0f} s")


if __name__ == "__main__":
    with RunLog("phase_1_evaluate"):
        main()
