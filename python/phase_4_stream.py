"""phase_4_stream.py - Build the per-second signal stream of the demo / measurement scenario (config.DEMO_SCENARIO).
Sources: contiguous runs of the subject's WESAD windows (raw wrist signals). Each condition has its own cursor,
so segments never reuse the same seconds (the calibration segment takes the first 5 min of baseline).
Kinds: clean | effort (HR x EFFORT_HR_FACTOR by time compression, EDA ramp to x EFFORT_EDA_GAIN over 30 s,
motion along gravity of strength DEMO_EFFORT_X x gate threshold) | fault_eda_off (EDA = 0 uS) | loss (no data).
Synthetic modifications are logic tests, NOT physiological models.
"""
import numpy as np

import config as C
from phase_3_replay_agreement import TAG, runs_of, seconds_of

FS = C.WRIST_FS


# For each condition, the longest stretch of continuous signal available for this subject.
def continuous_runs(npz):
    """{label: (bvp, eda, temp, acc) continuous arrays of the longest contiguous run of that label}."""
    t = npz["t_start"].astype(float)
    out = {}
    for run in runs_of(t):
        lab = int(npz["label"][run[0]])
        chunks = seconds_of(npz, run)
        arr = tuple(np.concatenate([c[k] for c in chunks]) for k in range(4))
        if lab not in out or len(arr[0]) > len(out[lab][0]):
            out[lab] = arr
    return out


# Synthetic physical effort (logic test only): faster pulse, rising skin conductance,
# and rhythmic arm motion strong enough to exceed the effort-gate threshold.
def _effort(seg, src_next_bvp, dur, thr):
    b, e, t, a = seg
    r = C.EFFORT_HR_FACTOR
    n = dur * FS["BVP"]
    bvp = np.interp(np.arange(n) * r, np.arange(len(src_next_bvp)), src_next_bvp)
    te = np.arange(len(e)) / FS["EDA"]
    eda = e * (1 + (C.EFFORT_EDA_GAIN - 1) * np.minimum(1.0, te / 30.0))
    g = a.mean(axis=0)
    g_dir = g / (np.linalg.norm(g) + 1e-9)
    amp = C.DEMO_EFFORT_X * thr / (4 * C.EFFORT_ACC_HZ)
    ta = np.arange(len(a)) / FS["ACC"]
    acc = np.clip(np.round(a + (amp / C.ACC_UNIT_G * np.sin(2 * np.pi * C.EFFORT_ACC_HZ * ta))[:, None] * g_dir), -128, 127)
    return bvp, eda, t, acc


# Builds the demo scenario of config.DEMO_SCENARIO as a list of segments of per-second signal.
def build_stream(subject, thr):
    """List of segments: dict(name, kind, seconds=[(bvp, eda, temp, acc) per second] or None for loss)."""
    npz = np.load(C.WIN_DIR / TAG / f"{subject}_windows.npz")
    runs = continuous_runs(npz)
    cursor = {lab: 0 for lab in runs}
    segs = []
    for name, kind, lab, dur in C.DEMO_SCENARIO:
        if kind == "loss":
            segs.append({"name": name, "kind": kind, "duration_s": dur, "seconds": None})
            continue
        b, e, t, a = runs[lab]
        s0 = cursor[lab]
        need = int(np.ceil(dur * (C.EFFORT_HR_FACTOR if kind == "effort" else 1))) + 1
        if (s0 + need) * FS["BVP"] > len(b):
            raise ValueError(f"{subject}: not enough signal for segment '{name}' (condition {lab})")
        sl = lambda x, ch, a0, a1: x[a0 * FS[ch]: a1 * FS[ch]]
        seg = (sl(b, "BVP", s0, s0 + dur), sl(e, "EDA", s0, s0 + dur), sl(t, "TEMP", s0, s0 + dur), sl(a, "ACC", s0, s0 + dur))
        if kind == "effort":
            seg = _effort(seg, sl(b, "BVP", s0, s0 + need), dur, thr)
            cursor[lab] += need
        else:
            cursor[lab] += dur
        # Sensor fault: skin-conductance electrode off the skin (reads 0).
        if kind == "fault_eda_off":
            seg = (seg[0], np.zeros_like(seg[1]), seg[2], seg[3])
        secs = [tuple(x[k * FS[ch]:(k + 1) * FS[ch]] for x, ch in zip(seg, ("BVP", "EDA", "TEMP", "ACC")))
                for k in range(dur)]
        segs.append({"name": name, "kind": kind, "duration_s": dur, "seconds": secs})
    return segs
