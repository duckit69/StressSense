"""config.py - Single source of truth for StressSense-HRC parameters (Python side).
Mirrored in firmware/sshrc/src/config.h: change both together.
Values marked [WESAD readme] come from data/WESAD/wesad_readme.pdf; [T] = design target.
"""
# Supervisor release: only the settings of the selected model are kept. The model comparisons
# (other models, window sweep, modality ablation) are archived in the full project.
from pathlib import Path

# --- Paths (relative to the repository root) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "WESAD"
LOG_DIR = ROOT / "logs"
WIN_DIR = ROOT / "data" / "windows"     # cut windows (npz), reused for training and replay
FEAT_DIR = ROOT / "data" / "features"   # one feature table (csv) per window setting

# --- Subjects present in WESAD (S1 and S12 discarded by the authors) [WESAD readme] ---
SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
            "S11", "S13", "S14", "S15", "S16", "S17"]

# --- Sampling rates in Hz [WESAD readme] ---
LABEL_FS = 700                     # labels are aligned with the chest RespiBAN at 700 Hz
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
ACC_UNIT_G = 1.0 / 64.0            # wrist ACC raw unit = 1/64 g [WESAD readme]

# --- Protocol labels [WESAD readme] ---
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement",
               4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}

# --- Windowing: fixed (chosen in the full project for its 30 s reaction time) ---
WINDOW_S = 30
HOP_S = 10

# Samples per window per channel (derived)
WIN_N = {ch: WINDOW_S * fs for ch, fs in WRIST_FS.items()}   # 30 s: ACC 960, BVP 1920, EDA 120, TEMP 120

# --- Window labelling (project decision, see dossier) ---
LABELS_USED = (1, 2, 3, 4)        # baseline, stress, amusement, meditation; 0 and 5-7 dropped
LABEL_PURITY = 1.0                # a window is kept only if all its labels are identical
TASK_BINARY = {1: 0, 3: 0, 2: 1}  # M1: non-stress (baseline + amusement) = 0, stress = 1

# --- Modalities (project decision) ---
CLASSIFIER_MODALITIES = ("BVP", "EDA", "TEMP")   # features fed to the stress classifier (F4/F5)
GATE_MODALITIES = ("ACC",)                       # activity index: quality check (F3) + effort gate (F6) only

# --- Feature extraction parameters (F4). Design choices to justify in the documents. ---
# Simple, causal-friendly algorithms so the C port (phase 3) can reproduce them exactly.
BVP_BAND_HZ = (0.7, 3.5)          # band-pass for beat detection (covers 42-210 bpm)
BVP_FILT_ORDER = 2                # Butterworth order (per band edge)
BVP_SKIP_S = 2.0                  # samples dropped at window start (filter transient)
BVP_PEAK_K = 0.5                  # peak must exceed K * std of the filtered window
IBI_MIN_S, IBI_MAX_S = 0.30, 1.50 # plausible inter-beat interval range (200-40 bpm)
IBI_MAX_DEV = 0.30                # reject IBI deviating > 30 % from the window median
EDA_SMOOTH_N = 4                  # moving average length (samples) = 1 s at 4 Hz
SCR_MIN_AMP_US = 0.01             # minimum rise for a phasic response (uS), to verify vs literature
SCR_MAX_RISE_S = 4.0              # max onset-to-peak time for one phasic response
# Column naming: <modality>_<name> are classifier candidates; q_<...> are quality metrics (F3),
# never fed to the classifier. Quality THRESHOLDS are set on phase 2.

# --- Selected classifier: logistic regression on per-operator calibrated features ---
# Hyperparameters fixed a priori (no tuning). Class imbalance handled by class_weight="balanced".
LOGREG_C = 1.0
LOGREG_MAX_ITER = 2000
IMPUTE_STRATEGY = "median"        # missing features filled with the training-fold median
N_JOBS = -1                       # test subjects run in parallel (results do not depend on it)
MODALITY_PREFIX = {"BVP": "bvp_", "EDA": "eda_", "TEMP": "temp_", "ACC": "acc_"}

# --- Per-operator calibration (F7b) ---
NORM_CALIB_S = 300                # calibration = first 5 min of the subject's baseline condition
# The calibration mean of each feature is subtracted from all of that subject's windows.
# Calibration windows are never used for training or scoring.
FROZEN_MODEL = "logreg"
FROZEN_NORM = True                # per-operator calibration mean subtracted (F7b)

# --- int8 quantisation of the frozen classifier (M3) ---
# Device chain: features -> minus calibration mean -> impute (training median) -> standardise (float)
#   -> int8 input (symmetric, per-tensor) . int8 weights (symmetric, per-tensor) + int32 bias -> sign = class
Q_INPUT_CLIP_PCT = 99.9           # input clip = this percentile of |z| on the TRAINING fold (no test data)
Q_BITS = 8                        # symmetric range [-127, 127]

# --- Signal quality (F3): fixed, physically motivated limits (not tuned on clean data) ---
Q_HR_MIN_BPM = 40                             # fewer beats than this rate -> BVP unusable
Q_BVP_MIN_BEATS = int(WINDOW_S * Q_HR_MIN_BPM / 60)
Q_BVP_MIN_VALID_FRAC = 0.5                    # at least half of the inter-beat intervals plausible
Q_BVP_MAX_IBI_CV = None                       # DISABLED: at 0.30 it flagged 29 % of clean baseline and 88 % of
                                              # stress windows (run 20261101-162954-582); q_bvp_ibi_cv still logged
Q_BVP_MAX_FLAT_S = 1.0                        # BVP identical for >= 1 s = stuck / disconnected
Q_EDA_MIN_US = 0.01                           # below = electrode contact lost (assumption, to verify)
Q_EDA_MAX_FLAT_S = 10.0                       # EDA identical for >= 10 s = stuck (assumption, to verify)
Q_EDA_FLAT_STD_US = 1e-6                      # (numerically) constant EDA = stuck sensor
Q_TEMP_MIN_C, Q_TEMP_MAX_C = 20.0, 42.0       # outside = sensor off the skin
Q_ACC_MAX_SAT_FRAC = 0.05                     # more saturated ACC samples -> effort gate input degraded
# Quality flag (2 bits): 0 = good, 1 = degraded (ACC only), 2 = bad (a classifier channel), 3 = no data (firmware)

# --- Effort gate (F6) and state mapping (F7) ---
GATE_ACT_PCT = 99.0               # gate threshold = this percentile of acc_activity on TRAINING-fold windows
GATE_REF_LABELS = (1, 4)          # ... of SEATED CALM conditions only (baseline, meditation). The first run used
                                  # all windows incl. stress -> threshold ~3 g/s, effort never gated (run 20261101-163141-852)
GATE_DECAY = 0.8                  # confidence of the held state x this, per hop, while effort lasts
CONF_MIN = 0.5                    # below -> robot side falls back (F10)
P_ELEV_ON, P_ELEV_OFF = 0.5, 0.4  # stress probability thresholds with hysteresis (design choice [T])
P_HIGH_ON, P_HIGH_OFF = 0.8, 0.7
N_DWELL = 2                       # consecutive windows needed before the state changes
STATE_NAMES = ["calm", "elevated", "high"]

# --- Synthetic scenarios (M11, M12): logic tests, NOT physiological validation ---
EFFORT_HR_FACTOR = 1.3            # BVP time-compressed -> HR x 1.3 (needs <= (WINDOW_S + HOP_S) / WINDOW_S)
EFFORT_EDA_GAIN = 1.3             # EDA ramps x1 -> x1.3 over the window (sweat)
EFFORT_ACC_HZ = 1.0               # periodic motion along gravity (v3; v1/v2 used 0.5 Hz on all axes, which
                                  # was orientation-dependent and below the gate: run 20261101-163757-034)
# --- Model export to the firmware (phase 2.4) ---
FW_SRC = ROOT / "firmware" / "sshrc" / "src"          # Arduino IDE sketch: firmware/sshrc/sshrc.ino + src/
EXPORT_HEADER = FW_SRC / "model_params.h"
N_GOLDEN = 50                     # reference windows for the C classifier unit test (phase 3)

# --- Laptop <-> ESP32 serial link (phase 3) ---
SERIAL_PORT = "COM5"
SERIAL_BAUD = 921600              # phase 3.2: raised from 115200 for fast replay (fallback: 115200 on both sides)
SERIAL_READY_TIMEOUT_S = 8        # the ESP32 resets when the port opens; wait for its READY line

# --- On-device replay (phase 3.2, M5-M7) ---
REPLAY_SUBJECTS = ["S2", "S10", "S15"]   # fixed before the run: one easy, one typical, one hard subject (M4 F1)
DEVICE_FEATURES = ["q_bvp_n_beats", "q_bvp_valid_frac", "q_bvp_ibi_cv", "q_bvp_flat_s", "bvp_hr", "bvp_sdnn",
                   "bvp_rmssd", "eda_mean", "eda_std", "eda_slope", "eda_scr_rate", "eda_scr_amp_sum", "q_eda_std",
                   "q_eda_min", "q_eda_max", "q_eda_flat_s", "temp_mean", "temp_std", "temp_slope", "q_temp_min",
                   "q_temp_max", "acc_mag_mean", "acc_mag_std", "acc_activity", "q_acc_sat_frac"]
FILTER_HEADER = FW_SRC / "bvp_filter.h"
AGREE_REL_TOL = 1e-3              # a device feature "agrees" if |dev - py| <= tol * max(1, |py|) [T]

# --- Phase 4: demo / measurement scenario (USB-only link, option A) ---
MEASURE_SUBJECT = "S2"            # M8-M10 measurement run: median subject by nested-LOSO macro-F1 (0.930), chosen
                                  # before any phase-4 result
DEMO_SUBJECT = "S6"               # live demo, chosen by a rule fixed on 2026-11-01 after the first phase-4 run (disclosed):
                                  # among subjects with LOSO F1 within +/-0.05 of the median, lowest
                                  # (stress quality-bad rate + stress gate rate) -> S6 (F1 0.896, 0.34)
DEMO_EXCLUDE = ["S2", "S6"]       # device model trained WITHOUT both, so measurement and demo are out-of-sample
DEMO_SPEED = 2.0                  # replay speed for the live demo (x real time); measurement run uses --speed 1
DEMO_SCENARIO = [                 # (name, kind, source condition, duration in signal seconds)
    ("calibration", "clean", 1, 300),       # first 5 min of baseline -> on-device calibration (fast-forward)
    ("baseline", "clean", 1, 60),
    ("effort", "effort", 1, 50),            # baseline + synthetic effort (X = DEMO_EFFORT_X): state held CALM
    ("stress", "clean", 2, 120),
    ("signal_loss", "loss", 0, 30),         # nothing sent: link/sensor loss
    ("sensor_fault", "fault_eda_off", 1, 30),
    ("recovery", "clean", 1, 70),           # 40 s was too short: buffer refill (30 s) + N_DWELL clean windows
]
DEMO_CALIB_FAST = True            # calibration segment streamed as fast as the link allows
DEMO_EFFORT_X = 2.0               # effort strength = X x gate threshold (M12 v3: gate fully effective at X >= 2)
ROBOT_UDP_PORT = 5005             # runner -> robot_sim on 127.0.0.1 (8-byte datagrams only)
ROBOT_ENVELOPE = 1.0              # speed limit set by the certified safety functions (read-only for the layer)
ROBOT_SPEED = {0: 1.0, 1: 0.7, 2: 0.4}   # ergonomic speed request per state [T design choice]
ROBOT_FALLBACK_SPEED = 0.25       # most conservative collaborative mode [T]
ROBOT_TIMEOUT_HOPS = 1.5          # no valid frame for 1.5 hops -> fallback (F10)
ROBOT_RECOVER_FRAMES = 2          # consecutive valid non-fallback frames needed to leave fallback
LED_PIN = 2                       # DevKit onboard LED (assumption)

# --- Reproducibility ---
SEED = 42
