"""serial_link.py - Open the ESP32 DevKit serial port without holding it in reset, and wait for READY.
DevKit auto-reset wiring: RTS -> EN (reset), DTR -> GPIO0 (boot mode). Opening a port from Python can
assert both lines, which holds the chip in reset or in download mode. So: open with both lines released,
pulse EN once for a clean boot, then wait for READY; if it was missed, ping ('P') until READY comes back.
"""
import time

import serial

import config as C


# Opens the ESP32 serial port, resets the board once and waits for its READY line.
# Returns (open port, READY text).
def open_device():
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = C.SERIAL_PORT, C.SERIAL_BAUD, 1
    ser.dtr = False                  # GPIO0 high -> normal boot, not download mode
    ser.rts = False                  # EN released
    ser.open()
    ser.rts = True                   # EN low: reset ...
    time.sleep(0.1)
    ser.rts = False                  # ... release -> clean boot
    t0 = time.time()
    while time.time() - t0 < C.SERIAL_READY_TIMEOUT_S:
        ln = ser.readline().decode(errors="replace").strip()
        if ln.startswith("READY"):
            return ser, ln
        if time.time() - t0 > 2:
            ser.write(b"P\n")            # READY missed: ask again
    ser.close()
    raise RuntimeError(f"no READY line from {C.SERIAL_PORT} within {C.SERIAL_READY_TIMEOUT_S} s: check the port, "
                       "close the Arduino Serial Monitor, and re-upload firmware/sshrc if it was changed")
