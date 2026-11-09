# StressSense-HRC: the selected stress model, from data to robot demo

A wrist-worn ESP32 estimates operator stress from wrist signals (BVP, EDA, skin temperature, acceleration) and sends only an
8-byte state frame to a simulated collaborative robot. The robot slows down for stress, not for physical effort, and never exceeds its certified
speed envelope.

This repository contains only the selected model and what is needed to reproduce its results:

| Choice | Value |
|---|---|
| Classifier | Logistic regression (12 parameters, int8 on the ESP32) |
| Input signals | BVP + EDA + skin temperature: 11 features. Acceleration is used only for the quality check and the effort gate. |
| Window | 30 s, updated every 10 s |
| Calibration | Per operator: the mean of the first 5 min of baseline is subtracted from every feature |

The comparison with other models, windows and signal sets that led to this choice is kept in the project archive, not here.

## What it reproduces

| Result | Script | Value in the original run |
|---|---|---|
| Stress detection, leave-one-subject-out over 15 WESAD subjects | `phase_1_evaluate.py` | macro-F1 0.907 ± 0.087 (SD); stress recall 88.1 %, false alarms 6.0 % |
| Final model on the ESP32: 2,755 training windows, gate threshold | `phase_2_export_model.py` | 1.324 g/s; int8 vs float agreement 99.85 % |
| ESP32 gives the same int32 result as Python on 50 test vectors | `phase_3_golden_test.py` | 50/50 identical, classifier 13 µs median |
| Demo: robot speed per scenario segment, privacy audit, fallback delays | `phase_4_demo_runner.py`, `phase_4_robot_sim.py`, `phase_4_analyse.py` | 8 B per frame, 0 violations; fallback 9.1–14.1 s |

Results come from Python 3.15.0, Windows 11, an ESP32-WROOM-32 DevKit and the package versions in `requirements.txt`. The random seed is fixed (`SEED = 42` in `python/config.py`).

## Folder layout

Scripts are numbered by project phase: phase 1 = data, features and evaluation; phase 2 = model export;
phase 3 = checks on the ESP32; phase 4 = robot demo and its analysis. Log file names (`SSHRC_D1_...`, `SSHRC_L2_...`) and
the run IDs quoted in comments keep the names they had in the original runs.

```
python/                        all scripts; every parameter is in python/config.py
  config.py                    single source of truth for all settings (mirrored in firmware/sshrc/src/config.h)
  runlog.py                    writes logs/console/, logs/config_snapshots/ and a row in logs/runs.csv for every run
  features.py, quality.py, calib.py, loso.py, quant.py, frame.py, serial_link.py   shared modules (imported, not run)
  phase_1_make_windows.py      cut 30 s windows (10 s hop) from WESAD
  phase_1_extract_features.py  25 features and quality metrics per window
  phase_1_evaluate.py          leave-one-subject-out test of the selected model (headline result)
  phase_2_export_model.py      train the final model, write firmware/sshrc/src/model_params.h
  phase_3_golden_test.py       check the int8 classifier on the ESP32
  phase_3_replay_agreement.py  replay helpers (also: device-vs-Python feature check)
  phase_4_stream.py            builds the demo scenario (imported by the runner)
  phase_4_demo_runner.py       streams the scenario to the ESP32, forwards frames to the robot
  phase_4_robot_sim.py         simulated robot and adaptation layer
  phase_4_analyse.py           measurements from the demo logs
firmware/sshrc/                Arduino sketch for the ESP32 (sshrc.ino + src/)
data/WESAD/                    put the WESAD dataset here (not included)
logs/                          created by the scripts (not committed)
```

## 1. Setup

### Python environment (venv)

Windows (PowerShell, from this folder):
```
py -3.15 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Dataset

Download WESAD from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection). Put the subject folders in `data/WESAD/` so that the files are `data/WESAD/S2/S2.pkl` … `data/WESAD/S17/S17.pkl` (15 subjects).

### ESP32 (only for steps 5–7)

- Arduino IDE with the ESP32 board package, board "ESP32 Dev Module".
- Set the serial port in `python/config.py`: `SERIAL_PORT = "COM5"`. Change it to your board's port.
- Close the Arduino Serial Monitor before running any Python script that talks to the board.

## 2. Run the experiment

Run every command from this folder, with the venv active. Each script prints its results and writes its logs to `logs/`.

**Offline part (no hardware, about 5 minutes):**
```
python python/phase_1_make_windows.py          # 1. cut the windows      -> data/windows/w30_h10/
python python/phase_1_extract_features.py      # 2. features             -> data/features/w30_h10.csv
python python/phase_1_evaluate.py              # 3. headline result      -> prints "LOSO over 15 subjects: macro-F1 = ..."
python python/phase_2_export_model.py          # 4. final model          -> firmware/sshrc/src/model_params.h
```
Step 4 trains on all 15 subjects. For an out-of-sample demo, where the demo subjects are not used in training, run `python python/phase_2_export_model.py --exclude S2 S6` instead.

**On the ESP32:**

5. Open `firmware/sshrc/sshrc.ino` in the Arduino IDE and upload it. Then close the Serial Monitor.
6. Run the golden test. Expect 50/50 class match:
   ```
   python python/phase_3_golden_test.py
   ```
7. Run the demo, in two terminals with the venv active:
   ```
   python python/phase_4_robot_sim.py --speed 2                 # terminal 1: robot window (start first)
   python python/phase_4_demo_runner.py --subject S6 --speed 2  # terminal 2: streams the scenario to the ESP32
   ```
   When the runner finishes, close the robot window. Then analyse the run:
   ```
   python python/phase_4_analyse.py --speed 2
   ```
