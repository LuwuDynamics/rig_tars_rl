#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tars_serial.py - PC side of the Tars serial log / test tool (firmware: main/boards/tars/tars_log*.c|cc).

  python3 tars_serial.py ports                       list serial ports
  python3 tars_serial.py check                       one-command health check (servos answer? voltage, IMU rate / gravity / bias)
  python3 tars_serial.py setup                       guided bring-up of the learned behaviours: IMU axes, servo directions, dry run
  python3 tars_serial.py scan                        ping the 3 servos, read the servo bus voltage
  python3 tars_serial.py watch                       live status line (Ctrl-C ends it)
  python3 tars_serial.py test hold 5                 robot still for 5 s: sensor noise / bias / voltage        -> saves + analyses
  python3 tars_serial.py test step 1 40              servo 1 step response, +-40 counts                          -> saves + analyses
  python3 tars_serial.py test sine 1 30 1.5 5        servo 1 sine, amplitude 30, 1.5 Hz, 5 cycles
  python3 tars_serial.py test walk 20 8              classic forward walk (20 counts) for 8 s while recording
  python3 tars_serial.py rec 20                      record 20 s of whatever the robot does (voice / BLE commands ...)
  python3 tars_serial.py --then "learn on go 20" rec 30   record 30 s and switch the learned mode on right after the recording started
  python3 tars_serial.py --seconds 14 --then "walk 20 8" watch 20   walk 8 s while the status line streams live (survives a power loss)
  python3 tars_serial.py term                       interactive terminal (type the device commands, dumps are saved)
  python3 tars_serial.py analyze FILE.tlog [--html report.html] [--csv flat.csv]

Any other first word is sent to the robot as a command line ("pose 0 0 30", "torque 0", "cal 0", "learn dry", ...); `help` lists them.
The port is found automatically (ESP32-S3 USB serial); use --port to choose.  Close idf.py monitor first - only one program can hold the port.
"""
import argparse
import datetime
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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
    def __init__(self, port, baud=115200, echo_raw=False):
        self.port = port
        self.baud = baud
        self.base_baud = baud
        self.q = queue.Queue()
        self.alive = True
        self.error = None
        self.echo_raw = echo_raw
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

    def go_fast(self, rate):
        """ask the robot to switch its UART to `rate`, follow it, confirm. Returns True on success; the old speed is restored on failure."""
        self.drain()
        self.send("baud %d" % rate)
        ack = None
        end = time.time() + 2.0
        while time.time() < end and ack is None:
            line = self.get(0.1)
            if line.startswith("@ok baud"):
                ack = line
            elif line.startswith("@err"):
                print(color("the robot refused the baud rate: %s" % line, "33"))
                return False
        if ack is None:
            return False
        old = self.baud
        try:
            time.sleep(0.05)
            self.set_local_baud(rate)
            time.sleep(0.15)
            self.ser.reset_input_buffer()
            self.drain()
            for _ in range(5):
                self.send("baud ok")
                end = time.time() + 0.6
                while time.time() < end:
                    if self.get(0.1).startswith("@ok baud confirmed"):
                        return True
        except (serial.SerialException, OSError) as e:
            print(color("this port cannot run at %d baud (%s)" % (rate, e), "33"))
        # not confirmed: the robot reverts to the old rate by itself after 8 s; go back and wait for it
        try:
            self.set_local_baud(old)
        except Exception:
            pass
        time.sleep(8.5)
        self.drain()
        return False

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

    def drain(self, seconds=0.0):
        end = time.time() + seconds
        n = 0
        while True:
            try:
                self.q.get(timeout=max(0.0, end - time.time()) if seconds else 0.0)
                n += 1
            except queue.Empty:
                break
        return n

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


def connect(args, stop_first=True):
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
            for b in (921600, 460800, 230400, 115200):           # left over from an earlier --fast session?
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
    except KeyboardInterrupt:
        dev.close()
        sys.exit(130)
    if hello is None:
        dev.close()
        sys.exit("The robot at %s does not answer 'hello'.\n"
                 "  - is the firmware with the log tool flashed (tars_log)?\n"
                 "  - is idf.py monitor or another program using the port?\n"
                 "  - the board may be stuck in download mode / a boot loop: press RESET and retry (or use --reset)" % port)
    if stop_first:
        dev.send("stop")                     # a previous session may have left something running
        time.sleep(0.2)
    dev.drain()
    print(color("connected: %s  %s" % (port, hello[len("@ok hello "):]), "32"))
    if args.fast:
        if dev.go_fast(args.fast_baud):
            print(color("fast link: %d baud" % dev.baud, "32"))
        else:
            print(color("could not switch to %d baud, staying at %d" % (args.fast_baud, dev.baud), "33"))
    return dev


# ---------------------------------------------------------------------------------------------------------------------
# capturing recordings
# ---------------------------------------------------------------------------------------------------------------------
def default_out_path(args, kind):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = args.out_dir
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s_%s.tlog" % (stamp, kind))


def finish_capture(args, lines, cmdline, path_hint=None):
    """write the captured dump lines to a .tlog file, print a quick analysis"""
    kind = "rec"
    for ln in lines:
        if ln.startswith("@begin dump"):
            for tok in ln.split():
                if tok.startswith("kind="):
                    kind = tok[5:]
            break
    path = path_hint or default_out_path(args, kind)
    args.last_capture = path
    with open(path, "w", encoding="utf-8") as f:
        f.write("# tars-log capture %s  command: %s  port: %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), cmdline, args.port or "auto"))
        for ln in lines:
            f.write(ln + "\n")
    print(color("saved %s  (%d lines)" % (path, len(lines)), "32"))
    if args.no_analyze:
        return path
    try:
        import tlog
        rec = tlog.load(path)
        print()
        print("\n".join(tlog.analyze(rec)))
        if args.html:
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(tlog.report_html(rec, title=os.path.basename(path)))
            print(color("report: %s" % args.html, "32"))
    except SystemExit as e:                  # numpy missing
        print(color(str(e), "33"))
    except Exception as e:                   # analysis must never lose the capture
        print(color("analysis failed (%s: %s) - the raw file is saved, send it to me" % (type(e).__name__, e), "31"))
    return path


def print_line(line, raw=False):
    if not line:
        return
    if line.startswith("@ok"):
        print(color(line, "32"))
    elif line.startswith("@err"):
        print(color(line, "31"))
    elif line.startswith("# w "):
        print(line[4:])
    elif line.startswith("#"):
        print(color(line, "36") if line.startswith("# [rl]") else line)
    elif line.startswith("@") and line[2:3] == ",":
        if raw:
            print(line)
    elif line.startswith("@begin") or line.startswith("@end") or line.startswith("@meta") or line.startswith("@fmt"):
        if raw:
            print(line)
    elif line.strip():
        print(color(line, "90"))             # boot messages / ESP_LOG text


def recover_recording(args, port, cmdline):
    """The USB device vanished while a recording was running (the servos' current steps can drop the USB link although the robot keeps running):
    wait for the port to come back, reconnect WITHOUT sending `stop` (a running recording must not be ended), wait until the recording has finished
    by itself and fetch it with `dump` (the data lives in the robot's RAM; it is lost if somebody switches the robot off meanwhile)."""
    print(color("the USB device disappeared (the robot keeps running): waiting for it to come back, then fetching the recording from the robot's RAM ...", "33"))
    t_end = time.time() + 240
    dev2 = None
    while time.time() < t_end and dev2 is None:
        time.sleep(1.0)
        if not os.path.exists(port):
            continue
        try:
            a2 = argparse.Namespace(**vars(args))
            a2.port = port
            dev2 = connect(a2, stop_first=False)
        except (SystemExit, Exception):                      # the node exists before the device is usable again: just try again
            dev2 = None
    if dev2 is None:
        print(color("could not reconnect; do NOT switch the robot off - reconnect and type `dump`", "31"))
        return 1
    try:
        t_wait = time.time() + 60
        while time.time() < t_wait:
            dev2.drain()
            dev2.send("status")
            lines = []
            t_try = time.time() + 1.5
            while time.time() < t_try:
                l = dev2.get(0.1)
                if l:
                    lines.append(l)
                    if l.startswith("@ok status"):
                        break
            mode_line = next((l for l in lines if l.startswith("# mode=")), "")
            if "mode=idle" in mode_line or "recording=0" in mode_line:
                break
            time.sleep(1.0)
        return run_command(dev2, args, "dump")
    finally:
        dev2.close()


def run_command(dev, args, cmdline):
    """send one command line and follow it to its end; recordings are saved + analysed"""
    words = cmdline.split()
    if not words:
        return 0
    first = words[0].lower()
    then = list(getattr(args, "then", None) or [])
    if then and first not in ("rec", "test", "watch", "live"):
        sys.exit("--then goes with rec / test / watch / live: it is sent right after that session has started")
    dev.drain()
    dev.send(cmdline)
    starter = first in ("rec", "test", "dump")
    streaming = first in ("watch", "live")
    expect_secs = 10.0
    if first == "rec" and len(words) > 1 and words[1].isdigit():
        expect_secs = int(words[1]) + 60.0
    elif first == "test":
        expect_secs = 150.0
        if len(words) > 4 and words[1] in ("walk", "turn") and words[4].isdigit():      # test walk <vx> <sec> <delay>
            expect_secs += int(words[4])
    elif first == "cal":
        expect_secs = 15.0
    elif streaming:
        expect_secs = 1e9                    # runs until Ctrl-C (or --seconds)
    capture = None
    started = False
    interrupted = False
    then_sent = False
    deadline = time.time() + expect_secs
    live_file = None
    t_start = time.time()
    status = 0
    try:
        while time.time() < deadline and dev.alive:
            if streaming and args.seconds and time.time() - t_start > args.seconds:
                break
            line = dev.get(0.2)
            if not line:
                continue
            if capture is not None:
                capture.append(line)
                if line.startswith("@end dump"):
                    continue
                if line.startswith("@ok") or line.startswith("@err"):
                    print_line(line)
                    break
                continue
            if line.startswith("@begin dump"):
                capture = [line]
                print(color("receiving the recording ...", "36"))
                deadline = time.time() + 120.0
                continue
            if streaming and first == "live" and line.startswith("@") and line[2:3] == ",":
                if live_file is None:
                    live_file = []
                live_file.append(line)
                continue
            print_line(line, args.raw)
            if line.startswith("@err"):
                status = 1
                if then_sent:                                   # a --then command was refused: end the recording (it still gets dumped), do not leave it running
                    print(color("a --then command failed: ending the recording", "33"))
                    dev.send("stop")
                    then_sent = False
                    continue
                break
            if line.startswith("@ok") and streaming and then and not then_sent and not started:
                started = True
                for c in then:                                  # watch / live is running (data streams to this PC in real time): now start what is to be watched
                    print(color("then: %s" % c, "36"))
                    dev.send(c)
                    time.sleep(0.05)
                then_sent = True
                continue
            if line.startswith("@ok") and not starter and not streaming:
                break
            if line.startswith("@ok") and starter and first == "dump":
                pass
            if line.startswith("@ok") and starter:
                if not started and then:
                    for c in then:                              # the recording runs: now start what is to be recorded
                        print(color("then: %s" % c, "36"))
                        dev.send(c)
                        time.sleep(0.05)
                    then_sent = True
                started = True
                deadline = time.time() + expect_secs
    except KeyboardInterrupt:
        interrupted = True
        print(color("\nCtrl-C: emergency stop (walking, learned mode, recording) ...", "33"))
        dev.send("estop")
        stop_seen = False
        t_end = time.time() + 40
        while time.time() < t_end:
            line = dev.get(0.2)
            if capture is not None:                           # the recording is arriving: collect it to its end
                if line:
                    capture.append(line)
                    if line.startswith("@ok") or line.startswith("@err"):
                        break
                continue
            if line.startswith("@begin dump"):
                capture = [line]
                continue
            if line.startswith("@ok estop") or line.startswith("@ok stop"):
                stop_seen = True
            # after `stop` a recording session still sends its data; anything else is finished with the @ok
            if stop_seen and not (starter and started):
                break
        streaming = False
    if streaming:
        dev.send("stop")
        time.sleep(0.4)
        while True:
            line = dev.get(0.2)
            if not line:
                break
            if first == "live" and line.startswith("@") and line[2:3] == ",":
                if live_file is None:
                    live_file = []
                live_file.append(line)
    if live_file:
        path = default_out_path(args, "live")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# tars-log live capture\n@begin dump kind=live\n")
            for ln in live_file:
                f.write(ln + "\n")
            f.write("@end dump\n")
        print(color("saved %s (%d events)" % (path, len(live_file)), "32"))
    if capture is not None:
        finish_capture(args, capture, cmdline, args.output)
    elif starter and started and first != "dump" and not dev.alive and not interrupted:
        status = recover_recording(args, dev.port, cmdline) or status
    elif starter and started and first != "dump":
        print(color("the robot did not send the recording (timeout) - do NOT switch it off: reconnect and type `dump` (the recording stays in its RAM)", "31"))
        status = 1
    return status


# ---------------------------------------------------------------------------------------------------------------------
# interactive terminal
# ---------------------------------------------------------------------------------------------------------------------
def cmd_term(args):
    dev = connect(args)
    try:
        import readline  # noqa: F401  (line editing + history)
    except ImportError:
        pass
    capture = []
    in_dump = [False]
    stop_flag = threading.Event()

    def printer():
        while not stop_flag.is_set():
            line = dev.get(0.2)
            if not line:
                if not dev.alive:
                    if dev.error:
                        print(color("\nserial port lost: %s" % dev.error, "31"))
                    stop_flag.set()
                continue
            if line.startswith("@begin dump"):
                in_dump[0] = True
                capture.clear()
                print(color("receiving the recording ...", "36"))
            if in_dump[0]:
                capture.append(line)
                if line.startswith("@end dump"):
                    in_dump[0] = False
                    finish_capture(args, list(capture), "(terminal)")
                    capture.clear()
                continue
            print_line(line, args.raw)

    th = threading.Thread(target=printer, daemon=True)
    th.start()
    print("Type device commands (help, scan, watch, test step 1 40, ...). Ctrl-C = stop, Ctrl-D / quit = leave.")
    try:
        while not stop_flag.is_set():
            try:
                text = input()
            except EOFError:
                break
            except KeyboardInterrupt:
                dev.send("estop")
                print(color("^C -> estop (walking, learned mode and recordings stop)", "33"))
                continue
            if text.strip().lower() in ("quit", "exit", "!q"):
                break
            if text.strip():
                dev.send(text)
    finally:
        try:
            dev.send("stop")
            time.sleep(0.2)
        except Exception:
            pass
        stop_flag.set()
        dev.close()
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# sub commands
# ---------------------------------------------------------------------------------------------------------------------
def cmd_ports(args):
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


def cmd_check(args):
    """one-command health check: servos answer, voltage, IMU rate / gravity / bias"""
    import tlog
    dev = connect(args)
    try:
        args.no_analyze = True
        args.last_capture = None
        print("1/2 scan ...")
        run_command(dev, args, "scan")
        print("2/2 keep the robot STILL and upright for 4 seconds ...")
        run_command(dev, args, "test hold 4")
        if not args.last_capture:
            print(color("no recording received", "31"))
            return 1
        rec = tlog.load(args.last_capture)
    finally:
        dev.close()
    marks = {"ok": color("[ ok ]", "32"), "warn": color("[warn]", "33"), "fail": color("[FAIL]", "31")}
    worst = "ok"
    print()
    for status, text in tlog.health_check(rec):
        print("%s %s" % (marks[status], text))
        if status == "fail" or (status == "warn" and worst == "ok"):
            worst = status
    print("\nrecording: %s" % args.last_capture)
    return 0 if worst != "fail" else 1


def cmd_setup(args, stages):
    """guided bring-up of the learned behaviours: IMU axes, servo directions, dry run"""
    import setup_flow
    dev = connect(args)
    try:
        return setup_flow.Setup(dev, setup_flow.ConsoleIO()).run(tuple(stages) if stages else ("pre", "imu", "map", "dry"))
    finally:
        dev.close()


def cmd_analyze(args):
    import tlog
    rec = tlog.load(args.file)
    print("\n".join(tlog.analyze(rec)))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(tlog.report_html(rec, title=os.path.basename(args.file)))
        print("report written: %s" % args.html)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            tlog.to_csv(rec, f)
        print("csv written: %s" % args.csv)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tars serial log / test tool (PC side)", add_help=True,
                                 epilog="Examples: tars_serial.py scan | watch | test hold 5 | test step 1 40 | term | analyze run.tlog --html run.html")
    ap.add_argument("--port", help="serial port (default: auto-detect the ESP32-S3)")
    ap.add_argument("--baud", type=int, default=115200, help="only matters for a UART bridge (default 115200)")
    ap.add_argument("--reset", action="store_true", help="reset the robot (RTS pulse) when connecting")
    ap.add_argument("--console", action="store_true", help="ask the robot to answer through the normal console path (slower, fallback)")
    ap.add_argument("--fast", action="store_true", help="switch the robot's UART to 921600 baud for big recordings (8x faster; restored when the tool exits)")
    ap.add_argument("--fast-baud", type=int, default=921600, help="rate used by --fast (default 921600)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "tars_logs"), help="where recordings are saved (default: tars_logs/ next to this script)")
    ap.add_argument("-o", "--output", help="file name for the recording of this command")
    ap.add_argument("--html", help="also write an HTML report with charts")
    ap.add_argument("--csv", help="analyze: write a flat CSV")
    ap.add_argument("--no-analyze", action="store_true", help="only save the recording")
    ap.add_argument("--raw", action="store_true", help="show data lines in the terminal")
    ap.add_argument("--seconds", type=float, default=0, help="watch/live: stop after this many seconds")
    ap.add_argument("--then", action="append", default=[], metavar="CMD",
                    help="rec/test: send this device command right after the recording has started, e.g. --then 'learn on go 20' (repeatable)")
    ap.add_argument("words", nargs="*", help="ports | term | analyze FILE | <device command ...>")
    args = ap.parse_intermixed_args(argv)
    words = args.words
    if not words:
        ap.print_help()
        return 0
    first = words[0].lower()
    if first == "ports":
        return cmd_ports(args)
    if first == "analyze":
        if len(words) < 2:
            sys.exit("usage: analyze FILE.tlog [--html out.html] [--csv out.csv]")
        args.file = words[1]
        return cmd_analyze(args)
    if first == "term":
        return cmd_term(args)
    if first == "check":
        return cmd_check(args)
    if first == "setup":
        names = {"pre": "pre", "preflight": "pre", "imu": "imu", "cal": "imu", "map": "map", "servo": "map", "dry": "dry"}
        stages = [names[w] for w in words[1:] if w in names]
        return cmd_setup(args, stages)
    dev = connect(args)
    try:
        return run_command(dev, args, " ".join(words))
    finally:
        dev.close()


def _on_terminate(signum, frame):
    raise KeyboardInterrupt            # the same emergency stop as Ctrl-C (sends `estop`), also when a launcher / the app's stop button ends this process


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
