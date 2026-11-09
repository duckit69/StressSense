"""phase_4_analyse.py - Phase 4 measurements from the runner + robot logs of the LAST scenario run.
M8  update period / jitter at the robot (frames inside paced segments, converted to signal time)
M9  privacy audit (from SSHRC_L4_privacy_x<speed>.csv, written by the runner)
M10 fallback latency: signal loss -> robot timeout fallback; sensor fault -> quality fallback; recovery time
Also: robot mode and applied speed per scenario segment (the demo's before/after).
Logs: logs/SSHRC_L4_frames_x<speed>.csv (M8), logs/SSHRC_L4_fallback_x<speed>.csv (M10),
      logs/SSHRC_L4_segments_x<speed>.csv
Usage (from the repository root):  python python/phase_4_analyse.py --speed 1   (measurement run)
                                   python python/phase_4_analyse.py --speed 2   (demo run)
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from runlog import RunLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=C.DEMO_SPEED)
    sfx = f"_x{ap.parse_args().speed:g}"
    # Inputs: the runner's timeline and privacy audit, and the robot's event log.
    sc = pd.read_csv(C.LOG_DIR / f"SSHRC_L4_scenario{sfx}.csv")
    rb = pd.read_csv(C.LOG_DIR / f"SSHRC_L4_robot{sfx}.csv")
    pv = pd.read_csv(C.LOG_DIR / f"SSHRC_L4_privacy{sfx}.csv").iloc[0]
    rb = rb[(rb["t_wall"] >= sc["wall_start"].min() - 2) & (rb["t_wall"] <= sc["wall_end"].max() + 30)]
    speed = float(pv["speed"])
    # M8: time between frames at the robot, converted to signal seconds.
    fr = rb[rb["event"] == "frame"].copy()
    if fr.empty:
        raise RuntimeError("robot log has no frames from this run: was phase_4_robot_sim.py running (and only once) "
                           "before the runner started?")
    seg_of = lambda t: next((r["segment"] for _, r in sc.iterrows() if r["wall_start"] <= t <= r["wall_end"] + 0.5), "after")
    fr["segment"] = fr["t_wall"].map(seg_of)
    fr["interval_wall_s"] = fr["t_wall"].diff()
    paced = sc.loc[sc["speed"] != "max", "segment"].tolist()
    ok = fr["segment"].isin(paced) & fr["segment"].eq(fr["segment"].shift()) & fr["segment"].ne("signal_loss")
    fr["interval_signal_s"] = fr["interval_wall_s"] * speed
    fr["used_for_m8"] = ok
    fr.to_csv(C.LOG_DIR / f"SSHRC_L4_frames{sfx}.csv", index=False)
    iv = fr.loc[ok, "interval_signal_s"]
    print(f"Run: subject {pv['subject']}, speed x{speed:g} | frames at robot {len(fr)}, lost "
          f"{int(rb['lost_total'].max()) if len(rb) else 0}, rejected {int((rb['event'] == 'reject').sum())}")
    print(f"M8 update period (signal time, {len(iv)} intervals): mean {iv.mean():.3f} s, SD {iv.std():.3f} s, "
          f"max |dev from {C.HOP_S} s| {np.max(np.abs(iv - C.HOP_S)):.3f} s  [T: {C.HOP_S} +/- 0.5 s]")
    print(f"M9 privacy: {int(pv['lines_frame'])} frames, {int(pv['frame_payload_bytes'])} B payload "
          f"({pv['bytes_per_frame']:.0f} B/frame), violations {int(pv['violations'])} -> {pv['verdict']}")

    # M10: delay from the start of signal loss / sensor fault to the robot's fallback,
    # and from the start of recovery to normal mode.
    md = rb[rb["event"] == "mode"]
    rows = []
    for name, cause_pat in (("signal_loss", "timeout"), ("sensor_fault", "frame: quality")):
        s = sc[sc["segment"] == name]
        if s.empty:
            continue
        t0 = float(s["wall_start"].iloc[0])
        hit = md[(md["t_wall"] >= t0) & (md["mode"] == "fallback") & md["cause"].str.startswith(cause_pat)]
        lat = (float(hit["t_wall"].iloc[0]) - t0) * speed if len(hit) else np.nan
        last_fr = fr[fr["t_wall"] < t0]["t_wall"].max()
        rows.append({"event": name, "trigger": cause_pat, "wall_start": t0, "latency_signal_s": lat,
                     "since_last_frame_signal_s": (float(hit["t_wall"].iloc[0]) - last_fr) * speed if len(hit) else np.nan,
                     "target": f"<= {2 * C.HOP_S} s [T]"})
    r = sc[sc["segment"] == "recovery"]
    if not r.empty:
        t0 = float(r["wall_start"].iloc[0])
        back = md[(md["t_wall"] >= t0) & (md["mode"] == "normal")]
        rows.append({"event": "recovery", "trigger": "valid frames", "wall_start": t0,
                     "latency_signal_s": (float(back["t_wall"].iloc[0]) - t0) * speed if len(back) else np.nan,
                     "since_last_frame_signal_s": np.nan,
                     "target": f"{C.ROBOT_RECOVER_FRAMES} valid frames after the 30 s buffer refill"})
    fb = pd.DataFrame(rows)
    fb.to_csv(C.LOG_DIR / f"SSHRC_L4_fallback{sfx}.csv", index=False)
    print("\nM10 fallback / recovery (signal seconds):")
    print(fb[["event", "trigger", "latency_signal_s", "since_last_frame_signal_s", "target"]].round(2).to_string(index=False))

    # Robot reaction per scenario segment: share of each state, fallback time and mean applied speed.
    seg = []
    for _, s in sc.iterrows():
        f = fr[fr["segment"] == s["segment"]]
        seg.append({"segment": s["segment"], "kind": s["kind"], "frames": len(f),
                    "calm_%": 100 * np.mean(f["state"] == 0) if len(f) else np.nan,
                    "elevated_%": 100 * np.mean(f["state"] == 1) if len(f) else np.nan,
                    "high_%": 100 * np.mean(f["state"] == 2) if len(f) else np.nan,
                    "frame_fallback_%": 100 * np.mean(f["fallback_flag"] == 1) if len(f) else np.nan,
                    "robot_fallback_%": 100 * np.mean(f["mode"] != "normal") if len(f) else np.nan,
                    "mean_applied_speed": f["applied_speed"].mean() if len(f) else np.nan,
                    "reasons": "; ".join(f"{k}={v}" for k, v in f["reason"].value_counts().items())})
    sg = pd.DataFrame(seg)
    sg.to_csv(C.LOG_DIR / f"SSHRC_L4_segments{sfx}.csv", index=False)
    print("\nRobot reaction per segment:")
    print(sg.drop(columns=["kind"]).round(2).to_string(index=False))


if __name__ == "__main__":
    with RunLog("phase_4_analyse"):
        main()
