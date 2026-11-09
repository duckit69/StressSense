"""loso.py - Leave-one-subject-out evaluation of the selected model (logistic regression).
Class balance: class_weight="balanced". No log I/O here; the runner writes the logs.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C


# The selected model. It is preceded by two steps that the ESP32 reproduces: fill missing
# features with the training median, then standardise each feature.
def make_model():
    clf = LogisticRegression(C=C.LOGREG_C, max_iter=C.LOGREG_MAX_ITER,
                             class_weight="balanced", random_state=C.SEED)
    return make_pipeline(SimpleImputer(strategy=C.IMPUTE_STRATEGY), StandardScaler(), clf)


# Confusion counts, accuracy and macro-F1 (mean F1 of the two classes) for one test subject.
def fold_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"n": len(y_true), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)}


# Seconds from the start of the stress condition to the first window predicted as stress.
def onset_latency(test_df, y_pred, win_s, hop_s):
    """W + k*H, k = stress windows before the first one predicted as stress (pure windows only)."""
    m = (test_df["y"].to_numpy() == 1)
    order = np.argsort(test_df["t_start"].to_numpy()[m])
    hits = np.flatnonzero(y_pred[m][order] == 1)
    return float(win_s + hits[0] * hop_s) if len(hits) else np.nan


# One leave-one-subject-out fold: train on the other subjects, test on `test_subject`.
def one_fold(df, cols, test_subject, train_subjects):
    tr = df[df["subject"].isin(train_subjects)]
    te = df[df["subject"] == test_subject]
    y_pred = make_model().fit(tr[cols], tr["y"]).predict(te[cols])
    met = fold_metrics(te["y"].to_numpy(), y_pred)
    met["n_imputed"] = int(te[cols].isna().sum().sum())
    return met, y_pred, te


# Full leave-one-subject-out: every subject is the test subject once.
def loso(df, cols, subjects):
    """{subject: (metrics, y_pred, test_df)} for LOSO restricted to `subjects`."""
    return {s: one_fold(df, cols, s, [x for x in subjects if x != s]) for s in subjects}
