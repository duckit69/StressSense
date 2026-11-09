"""phase_3_golden_test.py - Phase 3, step 1: run the 50 golden vectors through the int8 classifier ON THE ESP32.
Needs: firmware/sshrc uploaded with the Arduino IDE; Serial Monitor CLOSED (the port must be free).
Sends logs/SSHRC_L2_golden_classifier.csv line by line, reads the replies, compares with Python.
Log: logs/SSHRC_L3_golden.csv  (idx: expected/device accumulator, class, logit, |diff|, classify time in us)
Usage (from the repository root):  python python/phase_3_golden_test.py
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import numpy as np
import pandas as pd

import config as C
from runlog import RunLog
from serial_link import open_device


def main():
    # Golden vectors written by phase_2_export_model.py.
    g = pd.read_csv(C.LOG_DIR / "SSHRC_L2_golden_classifier.csv")
    cols = [c for c in g.columns if c.startswith(tuple(C.MODALITY_PREFIX[m] for m in C.CLASSIFIER_MODALITIES))]
    ser, ready = open_device()
    with ser:
        print(f"Device: {ready}")
        # The firmware must have been built with the same number of features.
        if f"n_feat={len(cols)}" not in ready:
            raise RuntimeError(f"feature count mismatch: Python {len(cols)} vs device line '{ready}'")
        rows = []
        for i, r in g.iterrows():
            vals = ",".join("nan" if pd.isna(v) else f"{v:.9g}" for v in r[cols])
            # Send one vector ('G' command) and read the device's int32 result and timing ('R' reply).
            ser.write(f"G,{i},{vals}\n".encode())
            ans = ser.readline().decode(errors="replace").strip()
            if not ans.startswith("R,"):
                raise RuntimeError(f"bad reply for {i}: '{ans}'")
            _, idx, acc, cls, logit, us = ans.split(",")
            rows.append({"idx": i, "subject": r["subject"], "t_start": r["t_start"],
                         "expected_acc": int(r["expected_acc_int32"]), "device_acc": int(acc),
                         "acc_absdiff": abs(int(acc) - int(r["expected_acc_int32"])),
                         "expected_class": int(r["expected_class"]), "device_class": int(cls),
                         "class_match": int(cls) == int(r["expected_class"]),
                         "expected_logit": float(r["expected_logit"]), "device_logit": float(logit),
                         "logit_absdiff": abs(float(logit) - float(r["expected_logit"])),
                         "classify_us": int(us)})
    # Compare device and Python results and save them.
    d = pd.DataFrame(rows)
    d.to_csv(C.LOG_DIR / "SSHRC_L3_golden.csv", index=False)
    print(f"Vectors: {len(d)} | class match: {int(d['class_match'].sum())}/{len(d)} | "
          f"accumulator exact: {int((d['acc_absdiff'] == 0).sum())}/{len(d)}, max |diff| {d['acc_absdiff'].max()} | "
          f"max |logit diff| {d['logit_absdiff'].max():.6f}")
    print(f"Classifier time on ESP32: median {np.median(d['classify_us']):.0f} us, max {d['classify_us'].max()} us")
    if not d["class_match"].all():
        print("MISMATCHED vectors: " + ", ".join(map(str, d.loc[~d["class_match"], "idx"])))


if __name__ == "__main__":
    with RunLog("phase_3_golden_test"):
        main()
