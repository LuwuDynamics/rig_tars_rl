#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read device status, servo replies, and firmware errors over the TARS serial port."""
import argparse
import math
import os
import queue
import signal
import sys
import threading
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    sys.exit("pyserial is missing:  pip install pyserial   (or run this with the ESP-IDF python: ~/.espressif/python_env/idf5.5_py3.11_env/bin/python)")

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if USE_COLOR else text


# ---------------------------------------------------------------------------------------------------------------------
# port handling
# ---------------------------------------------------------------------------------------------------------------------
def list_candidate_ports():
    ports = []
    for p in serial.tools.list_ports.comports():
        dev = p.device
        score = 0
        if p.vid == 0x303A:
            score += 10                      # Espressif USB Serial/JTAG
        elif p.vid in (0x1A86, 0x10C4, 0x0403):
            score += 8                       # WCH CH34x/CH9102, Silicon Labs CP210x, FTDI: the usual USB-UART bridges
        if "usbmodem" in dev or "ttyACM" in dev:
            score += 5
        if "usbserial" in dev or "ttyUSB" in dev or "wchusb" in dev or "SLAB" in dev:
            score += 3
        if dev.startswith("/dev/tty.") and os.path.exists(dev.replace("/dev/tty.", "/dev/cu.")):
            score -= 20                      # macOS: prefer the cu.* twin
        if score > 0:
            ports.append((score, dev, p))
    ports.sort(key=lambda x: -x[0])
    return ports


def find_port(preferred=None):
    if preferred:
        return preferred
    cands = list_candidate_ports()
    if not cands:
        all_ports = [p.device for p in serial.tools.list_ports.comports()]
        sys.exit("No ESP32 serial port found (plug in the USB cable; ports seen: %s). Use --port to choose one." % (", ".join(all_ports) or "none"))
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        sys.exit("Several serial ports match: %s\nChoose one with --port." % ", ".join(c[1] for c in cands))
    return cands[0][1]


# ---------------------------------------------------------------------------------------------------------------------
# device connection
# ---------------------------------------------------------------------------------------------------------------------
class Device:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.base_baud = baud
        self.q = queue.Queue()
        self.alive = True
        self.error = None
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.05
        self.ser.write_timeout = 2.0
        try:
            self.ser.exclusive = True        # a second program opening the port later fails loudly instead of silently stealing bytes
        except Exception:
            pass
        try:
            self.ser.open()                  # DTR/RTS stay as the OS sets them: no reset of the ESP32 (USB-Serial/JTAG)
        except (serial.SerialException, OSError) as e:
            sys.exit("Cannot open %s: %s\n(is idf.py monitor / another terminal still holding the port?)" % (port, e))
        self._buf = bytearray()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.alive:
            try:
                data = self.ser.read(max(1, self.ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                self.error = str(e)
                self.alive = False
                self.q.put(None)
                return
            if not data:
                continue
            self._buf += data
            while True:
                i = self._buf.find(b"\n")
                if i < 0:
                    break
                raw = bytes(self._buf[:i])
                del self._buf[:i + 1]
                self.q.put(raw.decode("utf-8", "replace").rstrip("\r"))
            if len(self._buf) > 200000:     # garbage without newlines
                self._buf.clear()

    def close(self):
        try:
            if self.baud != self.base_baud and self.alive:        # put the robot's UART back where the next program expects it
                self.send("baud %d" % self.base_baud)
                time.sleep(0.25)
                self.ser.baudrate = self.base_baud
                self.baud = self.base_baud
                time.sleep(0.1)
        except Exception:
            pass
        self.alive = False
        try:
            self.ser.close()
        except Exception:
            pass

    def set_local_baud(self, baud):
        self.ser.baudrate = baud
        self.baud = baud

    def hard_reset(self):
        """esptool style reset of an ESP32-S3 on its USB-Serial/JTAG port"""
        self.ser.rts = False
        self.ser.dtr = False
        time.sleep(0.05)
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False
        self.ser.dtr = False

    def send(self, text):
        self.ser.write((text.strip() + "\n").encode("utf-8"))
        self.ser.flush()

    def get(self, timeout=0.2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return ""

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def hello(self, timeout=6.0):
        """returns the '@ok hello ...' line or None"""
        end = time.time() + timeout
        while time.time() < end and self.alive:
            self.send("hello")
            t_try = time.time() + 0.5
            while time.time() < t_try:
                line = self.get(0.1)
                if line and line.startswith("@ok hello"):
                    return line
        return None


def connect(args):
    port = find_port(args.port)
    dev = Device(port, args.baud)
    hello = None
    try:
        if args.reset:
            dev.hard_reset()
            time.sleep(2.5)
        if args.console:
            dev.send("link console")
            time.sleep(0.3)
        # opening the port can reboot a board whose USB-UART bridge drives EN/IO0: give it time to come up
        hello = dev.hello(2.5)
        if hello is None:
            print(color("waiting for the robot (it may be rebooting because the port was opened) ...", "33"))
            hello = dev.hello(9.0)
        if hello is None:
            for b in (921600, 460800, 230400, 115200):           # the device can still be at a different rate
                if b == dev.baud:
                    continue
                try:
                    dev.set_local_baud(b)
                except Exception:
                    continue
                dev.drain()
                hello = dev.hello(1.5)
                if hello:
                    print(color("the robot answers at %d baud (left over from an earlier session)" % b, "33"))
                    break
            if hello is None:
                dev.set_local_baud(args.baud)
        if hello is None and not args.console:
            # the direct answer path may not work on this board: ask (blindly) for the slower console path
            print(color("no answer on the direct path - trying the console path (slower) ...", "33"))
            for _ in range(3):
                dev.send("link console")
                time.sleep(0.2)
            hello = dev.hello(5.0)
        if hello is None:
            sys.exit("The robot at %s does not answer 'hello'.\n"
                     "  - is the firmware with the log tool flashed (tars_log)?\n"
                     "  - is idf.py monitor or another program using the port?\n"
                     "  - the board may be stuck in download mode / a boot loop: press RESET and retry (or use --reset)" % port)
        dev.send("stop")  # Finish any previous serial session before starting a new one.
        time.sleep(0.2)
        dev.drain()
    except (KeyboardInterrupt, SystemExit, serial.SerialException, OSError):
        dev.close()
        raise
    print(color("connected: %s  %s" % (port, hello[len("@ok hello "):]), "32"))
    return dev


def print_line(line):
    if line.startswith("@ok"):
        print(color(line, "32"))
    elif line.startswith("@err"):
        print(color(line, "31"))
    elif line.startswith("# w "):
        print(line[4:], flush=True)
    else:
        print(line)


def wait_for_reply(dev, command):
    reply = f"@ok {command}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        line = dev.get()
        if not line:
            if not dev.alive:
                break
            continue
        print_line(line)
        if line.startswith("@err"):
            return 1
        if line == reply or line.startswith(reply + " "):
            return 0
    print(dev.error or f"Timed out waiting for {command}", file=sys.stderr)
    return 1


def run_command(dev, args):
    command = args.command
    request = f"watch {args.rate}" if args.rate is not None else command
    try:
        dev.send(request)
        status = wait_for_reply(dev, command)
        if status or command != "watch":
            return status

        deadline = time.monotonic() + (args.seconds or math.inf)
        while dev.alive and time.monotonic() < deadline:
            line = dev.get()
            if line:
                print_line(line)
        if not dev.alive:
            print(dev.error or "Serial port closed", file=sys.stderr)
            return 1
        dev.send("stop")
        return wait_for_reply(dev, "stop")
    except KeyboardInterrupt:
        if dev.alive:
            print(color("\nCtrl-C: sending emergency stop ...", "33"))
            dev.send("estop")
            wait_for_reply(dev, "estop")
        return 130


def cmd_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("no serial ports found")
    for p in sorted(ports, key=lambda x: x.device):
        vid = "%04X:%04X" % (p.vid, p.pid) if p.vid else "    -    "
        print("%-32s %s  %s" % (p.device, vid, p.description))
    cands = list_candidate_ports()
    if cands:
        print("\nwould use: %s" % cands[0][1])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="TARS serial diagnostics",
        epilog="Examples: tars_serial.py scan | tars_serial.py watch --seconds 5",
    )
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200, help="UART baud rate (default: 115200)")
    ap.add_argument("--reset", action="store_true", help="reset the robot when connecting")
    ap.add_argument("--console", action="store_true", help="use the firmware's console reply path")
    ap.add_argument("--seconds", type=float, help="duration of watch in seconds (0 means until interrupted)")
    ap.add_argument(
        "command", nargs="?", type=str.lower,
        choices=("ports", "scan", "watch", "status", "errors", "hello", "stop", "estop"),
    )
    ap.add_argument("rate", nargs="?", type=int, help="watch update rate (1 to 20 Hz)")
    args = ap.parse_intermixed_args(argv)
    if args.command is None:
        ap.print_help()
        return 0
    if args.rate is not None and (args.command != "watch" or not 1 <= args.rate <= 20):
        ap.error("only watch takes an update rate of 1 to 20 Hz")
    if args.seconds is not None and (args.command != "watch" or not math.isfinite(args.seconds) or args.seconds < 0):
        ap.error("--seconds requires watch and a nonnegative finite duration")
    if args.command == "ports":
        return cmd_ports()

    dev = connect(args)
    try:
        return run_command(dev, args)
    finally:
        dev.close()


def _on_terminate(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (serial.SerialException, OSError) as error:
        sys.exit(f"Serial connection failed ({error})")
