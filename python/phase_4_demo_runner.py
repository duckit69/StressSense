"""phase_4_demo_runner.py - Phase 4 (M8-M10 + live demo): replay the scenario to the ESP32 in OPERATIONAL mode,
audit EVERY line the device sends, forward only the 8-byte frames to the robot simulator (UDP, localhost).
Start robot_sim first (python python/phase_4_robot_sim.py), then this script in a second terminal.
Pacing: each second of signal is sent at t0 + k / speed (the calibration segment is fast-forwarded if
DEMO_CALIB_FAST); "loss" segments send nothing for duration / speed seconds.
Logs (suffix _x<speed>, e.g. _x1 for the measurement run, _x2 for the demo):
      logs/SSHRC_L4_device_tx_x<speed>.log  (every line received from the device, with wall time)
      logs/SSHRC_L4_privacy_x<speed>.csv    (M9 audit: line types, frame count, payload bytes, violations)
      logs/SSHRC_L4_scenario_x<speed>.csv   (segment timeline in wall time, for M8 / M10)
Usage (from the repository root):  python python/phase_4_demo_runner.py --subject S6            (demo, speed DEMO_SPEED)
                                   python python/phase_4_demo_runner.py --subject S2 --speed 1  (measurement run)
Needs: firmware/sshrc v4 uploaded, Serial Monitor closed.
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import argparse
import csv
import json
import re
import socket
import time

import numpy as np

import config as C
from phase_4_stream import build_stream
from runlog import RunLog
from serial_link import open_device

# In operational mode the device may send only these three kinds of line; anything else is
# counted as a privacy violation (measurement M9).
ALLOWED = [("ack", re.compile(r"^A$")), ("frame", re.compile(r"^D,[0-9a-f]{16}$")),
           ("control", re.compile(r"^OK,(session|operational)$"))]


# Returns which allowed kind a device line is, or 'VIOLATION'.
def classify_line(ln):
    for name, rx in ALLOWED:
        if rx.match(ln):
            return name
    return "VIOLATION"


# Sends one second of raw signal to the device and returns its reply.
def send_second(ser, c):
    b, e, t, a = c
    txt = ("S," + ",".join(f"{v:.9g}" for v in np.concatenate([b, e, t]).astype(float)) + ","
           + ",".join(str(int(v)) for v in np.asarray(a).ravel()))
    ser.write((txt + "\n").encode())
    return ser.readline().decode(errors="replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=C.DEMO_SPEED)
    ap.add_argument("--subject", default=C.DEMO_SUBJECT, choices=C.SUBJECTS)
    args = ap.parse_args()
    speed, subject = args.speed, args.subject
    sfx = f"_x{speed:g}"
    # The synthetic effort is scaled to the gate threshold of the exported model.
    thr = json.load(open(C.LOG_DIR / "SSHRC_L2_model_export.json"))["gate_threshold_gps"]
    segs = build_stream(subject, thr)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", C.ROBOT_UDP_PORT)
    tx = open(C.LOG_DIR / f"SSHRC_L4_device_tx{sfx}.log", "w", encoding="utf-8")
    counts = {"ack": 0, "frame": 0, "control": 0, "VIOLATION": 0}
    payload, timeline = 0, []

    # Logs every device line; in operational mode, forwards 8-byte frames (and nothing else) to the robot.
    def got(ln, phase):
        nonlocal payload
        kind = classify_line(ln) if phase == "operational" else "pre-operational"
        tx.write(f"{time.time():.3f}\t{phase}\t{kind}\t{ln}\n")
        if phase == "operational":
            counts[kind] += 1
            if kind == "frame":
                fr = bytes.fromhex(ln[2:])
                payload += len(fr)
                udp.sendto(fr, dest)                    # the ONLY thing forwarded to the robot
        return ln

    ser, ready = open_device()
    with ser:
        print(f"Device: {ready}")
        if "sshrc v4" not in ready:
            raise RuntimeError("firmware v4 expected (upload firmware/sshrc again)")
        excluded = ready.split("model_excluded=")[1].split()[0].split(",") if "model_excluded=" in ready else []
        if subject not in excluded:
            print(f"WARNING: {subject} was used to train the device model (excluded: {excluded}): NOT out-of-sample")
        got(ready, "pre-operational")
        # 'C' starts a new session (calibration restarts); 'O' switches the device to operational mode.
        ser.write(b"C\n"); got(ser.readline().decode(errors="replace").strip(), "operational")
        ser.write(b"O\n"); got(ser.readline().decode(errors="replace").strip(), "operational")
        print(f"Subject {subject} (out-of-sample: {subject in excluded}), speed x{speed:g}, segments: "
              + ", ".join(f"{s['name']}({s['duration_s']} s)" for s in segs))
        # Play the scenario segment by segment, pacing one second of signal every 1/speed seconds.
        for seg in segs:
            fast = seg["name"] == "calibration" and C.DEMO_CALIB_FAST
            t_start = time.time()
            if seg["kind"] == "loss":
                time.sleep(seg["duration_s"] / speed)
            else:
                for k, c in enumerate(seg["seconds"]):
                    if not fast:
                        wait = t_start + k / speed - time.time()
                        if wait > 0:
                            time.sleep(wait)
                    got(send_second(ser, c), "operational")
                if not fast:
                    wait = t_start + seg["duration_s"] / speed - time.time()
                    if wait > 0:
                        time.sleep(wait)
            t_end = time.time()
            timeline.append({"segment": seg["name"], "kind": seg["kind"], "duration_signal_s": seg["duration_s"],
                             "speed": "max" if fast else speed, "wall_start": round(t_start, 3),
                             "wall_end": round(t_end, 3)})
            print(f"  {seg['name']:13s} {seg['kind']:14s} {t_end - t_start:6.1f} s wall | frames so far {counts['frame']}")
    tx.close()
    with open(C.LOG_DIR / f"SSHRC_L4_scenario{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timeline[0].keys()))
        w.writeheader()
        w.writerows(timeline)
    total_bytes = sum(len(ln) for ln in open(C.LOG_DIR / f"SSHRC_L4_device_tx{sfx}.log", encoding="utf-8")
                      .read().splitlines() if ln.split("\t")[1] == "operational")
    # Privacy audit (M9): what left the device during the whole scenario.
    audit = {"subject": subject, "speed": speed, **{f"lines_{k}": v for k, v in counts.items()},
             "frame_payload_bytes": payload, "bytes_per_frame": payload / max(counts["frame"], 1),
             "operational_text_chars_total": total_bytes, "violations": counts["VIOLATION"],
             "verdict": "PASS: only acknowledgements, control replies and 8-byte frames left the device"
             if counts["VIOLATION"] == 0 else f"FAIL: see SSHRC_L4_device_tx{sfx}.log"}
    with open(C.LOG_DIR / f"SSHRC_L4_privacy{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit.keys()))
        w.writeheader()
        w.writerow(audit)
    print(f"\nM9 privacy audit: {counts['frame']} frames x 8 B = {payload} B of payload, "
          f"{counts['ack']} bare acks, {counts['control']} control replies, VIOLATIONS = {counts['VIOLATION']}")
    print(audit["verdict"])


if __name__ == "__main__":
    with RunLog("phase_4_demo_runner"):
        main()
