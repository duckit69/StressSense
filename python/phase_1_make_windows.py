"""phase_1_make_windows.py - Phase 1, step 1: cut pure 30 s windows every 10 s (config WINDOW_S / HOP_S) for every subject.
Grid starts at t = 0, 10, 20 ... s from the recording start (as the device would see it).
A window is kept only if all its 700 Hz labels are identical and in config.LABELS_USED.
Output per subject: data/windows/w<W>_h<H>/Sx_windows.npz  (bvp, eda, temp, acc, label, t_start)
Log: logs/SSHRC_D1_windows_w<W>_h<H>.csv  (subject, label_id, label_name, n_windows, acc_sat_frac)
Usage (from the repository root):  python python/phase_1_make_windows.py
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import argparse
import csv
import pickle

import numpy as np

import config as C
from runlog import RunLog


# Loads one WESAD subject file: the 700 Hz label track and the four wrist signals.
def load_subject(sid):
    with open(C.DATA_DIR / sid / f"{sid}.pkl", "rb") as f:   # trusted WESAD file
        d = pickle.load(f, encoding="latin1")
    labels = np.asarray(d["label"]).ravel().astype(np.int16)
    wrist = {ch: np.asarray(x, dtype=np.float32) for ch, x in d["signal"]["wrist"].items()}
    return labels, wrist


# Slides a window of win_s seconds every hop_s seconds over the recording and keeps
# the windows whose labels are all identical (pure) and of a condition we use.
def cut_subject(labels, wrist, win_s, hop_s):
    n_lab_win = win_s * C.LABEL_FS
    win_n = {ch: win_s * fs for ch, fs in C.WRIST_FS.items()}
    duration_s = len(labels) / C.LABEL_FS
    out = {ch: [] for ch in C.WRIST_FS}
    lab_out, t_out = [], []
    t = 0
    while t + win_s <= duration_s:
        seg = labels[t * C.LABEL_FS: t * C.LABEL_FS + n_lab_win]
        lab = int(seg[0])
        # A window is pure if every label sample inside it is the same.
        pure = np.mean(seg == lab) >= C.LABEL_PURITY
        if pure and lab in C.LABELS_USED:
            ok = True
            win = {}
            for ch, fs in C.WRIST_FS.items():
                s0 = t * fs
                w = wrist[ch][s0: s0 + win_n[ch]]
                if w.shape[0] != win_n[ch]:
                    ok = False
                    break
                # ACC keeps its 3 axes; the other channels are single columns.
                win[ch] = w if ch == "ACC" else w[:, 0]
            if ok:
                for ch in C.WRIST_FS:
                    out[ch].append(win[ch])
                lab_out.append(lab)
                t_out.append(t)
        t += hop_s

    arrays = {ch.lower(): np.stack(v) for ch, v in out.items()}
    arrays["label"] = np.asarray(lab_out, dtype=np.int8)
    arrays["t_start"] = np.asarray(t_out, dtype=np.float32)
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=C.SUBJECTS, choices=C.SUBJECTS)
    args = ap.parse_args()
    # The selected setting only: 30 s window, 10 s hop.
    configs = [(C.WINDOW_S, C.HOP_S)]
    for w, h in configs:
        if not 0 < h <= w:
            ap.error(f"need 0 < hop <= window (got {w}/{h})")
        print(f"Config w{w}_h{h}: overlap {100 * (1 - h / w):.0f}%")
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = {cfg: [] for cfg in configs}
    for sid in args.subjects:
        labels, wrist = load_subject(sid)
        for w, h in configs:
            tag = f"w{w}_h{h}"
            out_dir = C.WIN_DIR / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            # Cut and save this subject's windows, then log how many windows each condition gave.
            a = cut_subject(labels, wrist, w, h)
            np.savez_compressed(out_dir / f"{sid}_windows.npz", **a)
            line = []
            for lab in C.LABELS_USED:
                m = a["label"] == lab
                acc = a["acc"][m]
                sat = float(np.mean((acc <= -128) | (acc >= 127))) if acc.size else float("nan")
                rows[(w, h)].append([sid, lab, C.LABEL_NAMES[lab], int(m.sum()), f"{sat:.5f}"])
                line.append(f"{C.LABEL_NAMES[lab]}={int(m.sum())}")
            print(f"{sid:4s} {tag:8s} windows={len(a['label']):4d}  " + "  ".join(line))
        del labels, wrist

    for (w, h), r in rows.items():
        tag = f"w{w}_h{h}"
        log = C.LOG_DIR / f"SSHRC_D1_windows_{tag}.csv"
        with open(log, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["subject", "label_id", "label_name", "n_windows", "acc_sat_frac"])
            wr.writerows(r)
        tot = {lab: sum(x[3] for x in r if x[1] == lab) for lab in C.LABELS_USED}
        print(f"TOTAL {tag:8s} " + "  ".join(f"{C.LABEL_NAMES[k]}={v}" for k, v in tot.items())
              + f"   -> {log.name}")


if __name__ == "__main__":
    with RunLog("phase_1_make_windows"):
        main()
