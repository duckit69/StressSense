"""phase_4_robot_sim.py - Robot-side ergonomic adaptation layer (F9) + fallback (F10), simulated cobot cell.
Receives ONLY 8-byte state frames (UDP 127.0.0.1:ROBOT_UDP_PORT, sent by phase_4_demo_runner.py).
  request  = ROBOT_SPEED[state] in normal mode, ROBOT_FALLBACK_SPEED in fallback / waiting
  applied  = min(request, ROBOT_ENVELOPE)   (the certified safety envelope is read-only: the layer can only slow)
  fallback = frame says fallback (quality / effort low confidence / calibrating), or no valid frame for
             ROBOT_TIMEOUT_HOPS x HOP_S / speed seconds; normal again after ROBOT_RECOVER_FRAMES valid frames.
Rejected: datagrams that are not 8 bytes, bad CRC. Sequence gaps are counted as lost frames.
Log: logs/SSHRC_L4_robot_x<speed>.csv (every frame, mode change and rejection, with wall time)
Usage (from the repository root, start BEFORE the runner):  python python/phase_4_robot_sim.py [--speed 2]
                                                           python python/phase_4_robot_sim.py --headless --duration 900
Run record: logs/console/, logs/config_snapshots/, logs/runs.csv (see runlog.py).
"""
import argparse
import csv
import math
import socket
import time

import config as C
import frame as FR
from runlog import RunLog

# Display colours for calm / elevated / high.
COLORS = {0: "#2e7d32", 1: "#f9a825", 2: "#c62828"}


# The robot-side adaptation layer: checks each frame, chooses normal or fallback mode,
# and outputs the applied speed = min(speed request, certified envelope).
class Layer:
    def __init__(self, speed, log_path):
        self.timeout = C.ROBOT_TIMEOUT_HOPS * C.HOP_S / speed
        self.mode, self.cause = "waiting", "no frame yet"
        self.last_valid, self.last_seq, self.recover = None, None, 0
        self.n_frames, self.lost, self.rejected = 0, 0, 0
        self.last, self.last_raw = None, b""
        self.f = open(log_path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=["t_wall", "event", "seq", "state", "conf", "qflag", "effort",
                                                    "fallback_flag", "reason", "crc_ok", "mode", "cause",
                                                    "applied_speed", "lost_total", "frame_hex"])
        self.w.writeheader()

    # The layer can only lower the speed: the certified envelope is never exceeded or changed.
    def applied(self):
        req = C.ROBOT_SPEED[self.last["state"]] if (self.mode == "normal" and self.last) else C.ROBOT_FALLBACK_SPEED
        return min(req, C.ROBOT_ENVELOPE)

    def _log(self, event, **kw):
        self.w.writerow({"t_wall": f"{time.time():.3f}", "event": event, "mode": self.mode, "cause": self.cause,
                         "applied_speed": f"{self.applied():.2f}", "lost_total": self.lost, **kw})
        self.f.flush()

    def _set_mode(self, mode, cause):
        if mode != self.mode or cause != self.cause:
            self.mode, self.cause = mode, cause
            self._log("mode")

    # Handles one received datagram: reject wrong length or bad CRC, count lost frames,
    # then update the mode (fallback on the frame's fallback bit, normal after N good frames).
    def on_datagram(self, data):
        if len(data) != 8:
            self.rejected += 1
            self._log("reject", frame_hex=data[:32].hex(), reason=f"length {len(data)}")
            return
        d = FR.decode(data)
        if not d["crc_ok"]:
            self.rejected += 1
            self._log("reject", frame_hex=data.hex(), reason="crc")
            return
        if self.last_seq is not None and d["seq"] != (self.last_seq + 1) & 0xFFFF:
            self.lost += (d["seq"] - self.last_seq - 1) & 0xFFFF
        self.last_seq, self.last_valid, self.last, self.last_raw = d["seq"], time.time(), d, data
        self.n_frames += 1
        if d["fallback"]:
            self.recover = 0
            self._set_mode("fallback", f"frame: {d['reason']}")
        elif self.mode != "normal":
            self.recover += 1
            if self.recover >= C.ROBOT_RECOVER_FRAMES:
                self._set_mode("normal", "valid frames")
            else:
                self._set_mode("fallback", "recovering")
        self._log("frame", seq=d["seq"], state=d["state"], conf=f"{d['conf']:.3f}", qflag=d["qflag"],
                  effort=int(d["effort"]), fallback_flag=int(d["fallback"]), reason=d["reason"], crc_ok=1,
                  frame_hex=data.hex())

    # Fallback if no valid frame has arrived for 1.5 update periods (link or sensor loss).
    def watchdog(self):
        if self.last_valid is not None and self.mode != "waiting" and time.time() - self.last_valid > self.timeout:
            if self.cause != "timeout":
                self.recover = 0
                self._set_mode("fallback", "timeout")


# Simple Tkinter window: a 2-link arm moving at the applied speed, a speed gauge with the
# envelope line, and the last frame received.
def gui(layer, sock):
    import tkinter as tk
    root = tk.Tk()
    root.title("StressSense-HRC - robot side (simulated cobot cell)")
    cv = tk.Canvas(root, width=900, height=520, bg="white")
    cv.pack()
    phase = [0.0]

    def tick():
        try:
            while True:
                layer.on_datagram(sock.recv(64))
        except BlockingIOError:
            pass
        layer.watchdog()
        sp = layer.applied()
        phase[0] += sp * 0.12
        cv.delete("all")
        # cobot arm (2 links), pick-and-place motion at the applied speed
        bx, by = 220, 420
        a1 = math.radians(-60 + 40 * math.sin(phase[0]))
        a2 = math.radians(70 + 30 * math.sin(phase[0] * 1.3))
        x1, y1 = bx + 150 * math.cos(a1), by + 150 * math.sin(a1)
        x2, y2 = x1 + 120 * math.cos(a1 + a2), y1 + 120 * math.sin(a1 + a2)
        cv.create_rectangle(bx - 40, by, bx + 40, by + 30, fill="#555")
        cv.create_line(bx, by, x1, y1, width=18, fill="#1565c0", capstyle="round")
        cv.create_line(x1, y1, x2, y2, width=14, fill="#42a5f5", capstyle="round")
        cv.create_oval(x2 - 10, y2 - 10, x2 + 10, y2 + 10, fill="#ffb300")
        cv.create_text(220, 480, text="simulated cobot (motion speed = applied speed)", fill="#555")
        # speed gauge with the certified envelope
        gx, gy, gh = 470, 80, 300
        cv.create_text(gx + 30, 55, text="speed", font=("Arial", 12, "bold"))
        cv.create_rectangle(gx, gy, gx + 60, gy + gh, outline="#333")
        cv.create_rectangle(gx, gy + gh * (1 - sp), gx + 60, gy + gh, fill="#1565c0", outline="")
        ey = gy + gh * (1 - C.ROBOT_ENVELOPE)
        cv.create_line(gx - 10, ey, gx + 70, ey, fill="#c62828", width=3, dash=(6, 3))
        cv.create_text(gx + 30, gy + gh + 20, text=f"{100 * sp:.0f} %", font=("Arial", 14, "bold"))
        cv.create_text(gx + 30, ey - 12, text="safety envelope (certified, read-only)", fill="#c62828",
                       font=("Arial", 8))
        # state panel
        px = 590
        st = layer.last["state"] if layer.last else None
        cv.create_text(px, 80, anchor="w", text="operator state", font=("Arial", 11))
        cv.create_text(px, 115, anchor="w", font=("Arial", 26, "bold"), fill=COLORS.get(st, "#777"),
                       text=C.STATE_NAMES[st].upper() if st is not None else "-")
        info = [f"confidence  {layer.last['conf']:.2f}" if layer.last else "confidence  -",
                f"reason      {layer.last['reason']}" if layer.last else "reason      -",
                f"robot mode  {layer.mode} ({layer.cause})",
                f"frames      {layer.n_frames} x 8 B = {8 * layer.n_frames} B received",
                f"lost {layer.lost} | rejected {layer.rejected}",
                f"last frame  {layer.last_valid and time.time() - layer.last_valid or 0:.1f} s ago"]
        for i, t in enumerate(info):
            cv.create_text(px, 165 + 26 * i, anchor="w", text=t, font=("Consolas", 11))
        if layer.last:
            cv.create_text(px, 340, anchor="w", font=("Consolas", 11), fill="#333",
                           text="the ONLY data from the wearable:")
            cv.create_text(px, 365, anchor="w", font=("Consolas", 14, "bold"),
                           text=" ".join(f"{b:02x}" for b in layer.last_raw))
        if layer.mode != "normal":
            cv.create_rectangle(0, 0, 900, 40, fill="#c62828" if layer.mode == "fallback" else "#777", outline="")
            cv.create_text(450, 20, fill="white", font=("Arial", 16, "bold"),
                           text=f"FALLBACK - most conservative collaborative mode ({layer.cause})"
                           if layer.mode == "fallback" else "WAITING FOR THE WEARABLE")
        root.after(40, tick)

    root.after(40, tick)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=C.DEMO_SPEED, help="must match the runner's --speed")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=0, help="headless: stop after this many seconds")
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Listen for frames from phase_4_demo_runner.py on this computer only.
    sock.bind(("127.0.0.1", C.ROBOT_UDP_PORT))
    sock.setblocking(False)
    layer = Layer(a.speed, C.LOG_DIR / f"SSHRC_L4_robot_x{a.speed:g}.csv")
    print(f"Robot layer listening on 127.0.0.1:{C.ROBOT_UDP_PORT} | speed x{a.speed:g} | timeout {layer.timeout:.1f} s "
          f"| envelope {C.ROBOT_ENVELOPE} | speeds {C.ROBOT_SPEED} fallback {C.ROBOT_FALLBACK_SPEED}")
    try:
        if a.headless:
            t0 = time.time()
            while not a.duration or time.time() - t0 < a.duration:
                try:
                    while True:
                        layer.on_datagram(sock.recv(64))
                except BlockingIOError:
                    pass
                layer.watchdog()
                time.sleep(0.02)
        else:
            gui(layer, sock)
    except KeyboardInterrupt:
        pass
    finally:
        layer.f.close()
        print(f"Frames {layer.n_frames}, lost {layer.lost}, rejected {layer.rejected} -> logs/SSHRC_L4_robot_x{a.speed:g}.csv")


if __name__ == "__main__":
    with RunLog("phase_4_robot_sim"):
        main()
