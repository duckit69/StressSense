"""frame.py - The 8-byte state frame (F8): the ONLY data that leaves the worn device.
byte 0: version(2 bits, =1) | state(2) | quality flag(2) | effort(1) | fallback(1)
byte 1: confidence, 0..255 (= round(conf * 255))
byte 2-3: sequence number, uint16 little-endian (loss / staleness detection on the robot side)
byte 4: reason code (0 normal, 1 effort hold, 2 effort low confidence, 3 quality, 4 calibrating)
byte 5: reserved (0)
byte 6-7: CRC-16/CCITT-FALSE of bytes 0-5, little-endian
Mirrors firmware/sshrc/src/decision.cpp:pack().
"""
# Reason codes carried in byte 4 of the frame.
REASONS = ["normal", "effort_hold", "effort_low_conf", "quality", "calibrating"]


# CRC-16/CCITT-FALSE checksum (polynomial 0x1021, start 0xFFFF): detects corrupted frames.
def crc16_ccitt(data):
    c = 0xFFFF
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


# Builds the 8-byte frame (same layout as the firmware).
def encode(state, qflag, effort, fallback, conf, seq, reason):
    b0 = (1 << 6) | (state << 4) | (qflag << 2) | (int(effort) << 1) | int(fallback)
    body = bytes([b0, int(round(min(max(conf, 0.0), 1.0) * 255)), seq & 0xFF, (seq >> 8) & 0xFF, reason, 0])
    c = crc16_ccitt(body)
    return body + bytes([c & 0xFF, c >> 8])


# Reads an 8-byte frame back into its fields and checks the CRC.
def decode(fr):
    if len(fr) != 8:
        raise ValueError("frame must be 8 bytes")
    ok = crc16_ccitt(fr[:6]) == (fr[6] | fr[7] << 8)
    b0 = fr[0]
    return {"crc_ok": ok, "version": b0 >> 6, "state": (b0 >> 4) & 3, "qflag": (b0 >> 2) & 3,
            "effort": bool(b0 & 2), "fallback": bool(b0 & 1), "conf": fr[1] / 255.0,
            "seq": fr[2] | fr[3] << 8, "reason": REASONS[fr[4]] if fr[4] < len(REASONS) else fr[4]}
