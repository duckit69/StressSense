"""phase_3_replay_agreement.py - Phase 3.2 (M5, M6, M7): replay raw WESAD signals to the ESP32, which buffers
30 s, extracts the 25 features + quality metrics on-device every 10 s; compare with Python (features.py).
Subjects: config.REPLAY_SUBJECTS. Each contiguous run of windows is streamed second by second
(first window's 30 s, then the last 10 s of each next window), so device windows = Python windows.
Class check: both feature sets go through the exported int8 model with the SAME calibration means
(from the Python features), so only feature differences can change the class.
Logs: logs/SSHRC_L3_agreement.csv  (window: py_<feature>, dev_<feature>, quality flags, classes)
      logs/SSHRC_L3_timing.csv     (window: per-stage device time in us)
      logs/SSHRC_L3_memory.txt     (device memory after the replay)
Usage (from the repository root):  python python/phase_3_replay_agreement.py
Needs: firmware/sshrc (v2) uploaded, Serial Monitor closed.
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import json
import time

import numpy as np
import pandas as pd

import config as C
from calib import calibration_means, feature_cols
from quality import quality_flag
from quant import int8_logit
from runlog import RunLog
from serial_link import open_device

TAG = f"w{C.WINDOW_S}_h{C.HOP_S}"
FS = C.WRIST_FS
# Per-stage timings the firmware reports for each window (microseconds).
TIMES = ["filter_us", "beats_us", "eda_us", "temp_us", "acc_us", "total_us", "window_us"]


# Groups windows into runs of consecutive windows (start times exactly HOP_S apart).
def runs_of(t):
    """Split window indices (time order) into contiguous runs (step HOP_S)."""
    order = np.argsort(t)
    out, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if abs(t[b] - t[a] - C.HOP_S) < 1e-6:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


# Turns a run of overlapping windows back into the continuous signal, one second at a time,
# which is how the device receives it.
def seconds_of(npz, run):
    """Per-second (bvp, eda, temp, acc) chunks for a contiguous run; checks the overlaps are identical."""
    W, H = C.WINDOW_S, C.HOP_S
    # Load each signal array once: an .npz file re-reads the whole array on every npz[ch] access, and every
    # slice would then keep its own full copy alive (several GB for a 25-minute baseline).
    arr = {ch: npz[ch] for ch in ("bvp", "eda", "temp", "acc")}
    sl = lambda ch, i, s0, s1: arr[ch][i][s0 * FS[ch.upper()]: s1 * FS[ch.upper()]]
    chunks = []
    for k, i in enumerate(run):
        s_from = 0 if k == 0 else W - H
        if k:
            prev = run[k - 1]
            assert np.array_equal(sl("bvp", i, 0, W - H), sl("bvp", prev, H, W)), "windows do not overlap as expected"
        for s in range(s_from, W):
            chunks.append(tuple(sl(ch, i, s, s + 1) for ch in ("bvp", "eda", "temp", "acc")))
    return chunks


# Sends one second of raw signal ('S' command) and returns the device's reply line.
def send_second(ser, c):
    b, e, t, a = c
    acc = np.asarray(a).ravel()
    if not np.all(acc == np.round(acc)):
        raise ValueError("ACC samples are not integers")
    txt = ("S," + ",".join(f"{v:.9g}" for v in np.concatenate([b, e, t]).astype(float)) + ","
           + ",".join(str(int(v)) for v in acc))
    ser.write((txt + "\n").encode())
    return ser.readline().decode(errors="replace").strip()


# Reads the exported int8 model parameters (logs/SSHRC_L2_model_export.json).
def load_q():
    e = json.load(open(C.LOG_DIR / "SSHRC_L2_model_export.json"))
    q = {k: np.asarray(e[k]) for k in ("median", "mean", "scale")}
    q.update({"s_x": e["s_x"], "s_w": e["s_w"], "qw": np.asarray(e["qw"], dtype=np.int8), "qb": int(e["qb"])})
    return q, e["features"]


def main():
    q, cls_cols = load_q()
    ref = pd.read_csv(C.FEAT_DIR / f"{TAG}.csv")
    # Calibration means from the Python features, used for BOTH feature sets in the class check.
    mu = calibration_means(ref, feature_cols(ref, C.MODALITY_PREFIX.keys()))
    ser, ready = open_device()
    rows, trows = [], []
    with ser:
        print(f"Device: {ready}")
        # Check the device computes the same 25 features in the same order as config.DEVICE_FEATURES.
        ser.write(b"N\n")
        names = ser.readline().decode(errors="replace").strip().split(",")
        if names[0] != "NAMES" or names[1:] != C.DEVICE_FEATURES:
            raise RuntimeError(f"device feature list differs from config.DEVICE_FEATURES: {names}")
        t0 = time.time()
        for sid in C.REPLAY_SUBJECTS:
            npz = np.load(C.WIN_DIR / TAG / f"{sid}_windows.npz")
            tt = npz["t_start"].astype(float)
            rs = runs_of(tt)
            for ri, run in enumerate(rs):
                # Empty the device's 30 s buffer before each run, so its windows line up with Python's.
                ser.write(b"R\n")
                if ser.readline().decode(errors="replace").strip() != "OK,reset":
                    raise RuntimeError("no reset acknowledgement")
                k = 0
                for c in seconds_of(npz, run):
                    ans = send_second(ser, c)
                    if ans.startswith("A,"):
                        continue
                    if not ans.startswith("F,"):
                        raise RuntimeError(f"{sid} run {ri}: unexpected reply '{ans[:80]}'")
                    v = ans.split(",")
                    dev = dict(zip(C.DEVICE_FEATURES, map(float, v[2:2 + len(C.DEVICE_FEATURES)])))
                    tms = dict(zip(TIMES, map(int, v[2 + len(C.DEVICE_FEATURES):2 + len(C.DEVICE_FEATURES) + len(TIMES)])))
                    i = run[k]
                    k += 1
                    py = ref[(ref["subject"] == sid) & np.isclose(ref["t_start"], tt[i])].iloc[0]
                    xp = (py[cls_cols].astype(float) - mu[sid][cls_cols]).to_numpy()[None, :]
                    xd = (pd.Series(dev)[cls_cols] - mu[sid][cls_cols]).to_numpy()[None, :]
                    cp, cd = int8_logit(q, xp)[1][0], int8_logit(q, xd)[1][0]
                    rows.append({"subject": sid, "t_start": tt[i], "label": int(npz["label"][i]),
                                 **{f"py_{n}": float(py[n]) for n in C.DEVICE_FEATURES},
                                 **{f"dev_{n}": dev[n] for n in C.DEVICE_FEATURES},
                                 "py_qflag": int(quality_flag({n: [py[n]] for n in C.DEVICE_FEATURES})[0]),
                                 "dev_qflag": int(quality_flag({n: [dev[n]] for n in C.DEVICE_FEATURES})[0]),
                                 "py_class": int(cp), "dev_class": int(cd)})
                    trows.append({"subject": sid, "t_start": tt[i], **tms})
                print(f"  {sid} run {ri + 1}/{len(rs)}: {k} windows ({time.time() - t0:.0f} s elapsed)")
        ser.write(b"M\n")
        mem = ser.readline().decode(errors="replace").strip()

    # Save the comparison and print per-feature agreement.
    d = pd.DataFrame(rows)
    d.to_csv(C.LOG_DIR / "SSHRC_L3_agreement.csv", index=False)
    pd.DataFrame(trows).to_csv(C.LOG_DIR / "SSHRC_L3_timing.csv", index=False)
    (C.LOG_DIR / "SSHRC_L3_memory.txt").write_text(f"{ready}\n{mem}\n", encoding="utf-8")

    print(f"\nWindows compared: {len(d)} | memory after replay: {mem}")
    print(f"{'feature':18s} {'agree %':>8s} {'max |diff|':>11s} {'NaN py/dev':>10s}")
    for n in C.DEVICE_FEATURES:
        a, b = d[f"py_{n}"].to_numpy(), d[f"dev_{n}"].to_numpy()
        both = np.isnan(a) & np.isnan(b)
        ok = both | (np.abs(a - b) <= C.AGREE_REL_TOL * np.maximum(1, np.abs(a)))
        mx = np.nanmax(np.abs(a - b)) if np.any(~np.isnan(a - b)) else 0.0
        print(f"{n:18s} {100 * ok.mean():8.2f} {mx:11.3g} {f'{np.isnan(a).sum()}/{np.isnan(b).sum()}':>10s}")
    print(f"\nQuality flag identical: {100 * np.mean(d['py_qflag'] == d['dev_qflag']):.2f} % | "
          f"class identical: {100 * np.mean(d['py_class'] == d['dev_class']):.2f} % [T >= 99 %]")
    tm = pd.DataFrame(trows)
    print("Device time per window (us): " + ", ".join(f"{c} median {tm[c].median():.0f} / max {tm[c].max()}"
                                                       for c in ("filter_us", "beats_us", "eda_us", "total_us", "window_us")))


if __name__ == "__main__":
    with RunLog("phase_3_replay_agreement"):
        main()
