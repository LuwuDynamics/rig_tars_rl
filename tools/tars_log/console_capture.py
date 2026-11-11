#!/usr/bin/env python3
"""Passive console recorder: keep the serial port open WITHOUT touching DTR / RTS (no reset: forcing them low resets an ESP32-S3) and append everything the robot prints, with a PC time stamp, to a file.
Use it to catch the register dump and backtrace the robot prints before it reboots after a crash (rst=PANIC), which no other tool sees while it is idle.
While it runs it owns the port: stop it (Ctrl-C / SIGTERM) before using tars_serial.py.

usage: python3 console_capture.py [--port /dev/cu.usbmodemXXXX] [--baud 115200] [--out tars_logs/console_YYYYMMDD_HHMMSS.log]"""
import argparse
import os
import signal
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is missing:  pip install pyserial   (or run this with the ESP-IDF python)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tars_serial import find_port    # noqa: E402   the same port auto-detection as the other tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.port = find_port(a.port or None)
    out = a.out or os.path.join(HERE, "tars_logs", "console_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = a.port, a.baud, 0.5
    try:
        s.exclusive = True        # a second program opening the port fails loudly instead of stealing bytes
    except Exception:
        pass
    s.open()                      # DTR / RTS stay as the OS sets them (forcing them low resets an ESP32-S3 on its USB-Serial/JTAG port)
    f = open(out, "a", buffering=1)
    f.write("# console capture started %s on %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), a.port))
    buf = b""
    while not stop:
        try:
            d = s.read(4096)
        except Exception as e:                       # the device vanished (cable): note it and try to come back
            f.write("# %s port error: %s\n" % (time.strftime("%H:%M:%S"), e))
            time.sleep(1.0)
            try:
                s.close()
                s.open()
                f.write("# %s port reopened\n" % time.strftime("%H:%M:%S"))
            except Exception:
                pass
            continue
        if not d:
            continue
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            f.write("%s %s\n" % (time.strftime("%H:%M:%S"), line.decode("utf-8", "replace").rstrip("\r")))
    f.write("# console capture stopped %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    f.close()
    s.close()


if __name__ == "__main__":
    main()
