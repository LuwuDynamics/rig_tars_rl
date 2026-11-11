#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end test of the Tars log tool on the host:

   tars_serial.py / tlog.py (the real PC tool)  <-- pseudo terminal -->  tars_log_sim (the REAL firmware shell + core, simulated robot)

Run:  make -C tools/tars_log/test test      (or: python3 test_protocol.py after `make`)
"""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tars_serial  # noqa: E402
import tlog  # noqa: E402

SIM = os.environ.get("SIM_BIN") or os.path.join(HERE, "tars_log_sim")
OUT = tempfile.mkdtemp(prefix="tars_log_test_")


class Sim:
    def __init__(self, **env):
        e = dict(os.environ)
        e.setdefault("SIM_SPEED", "6")
        e.setdefault("SIM_RING", "30000")
        e.update({k: str(v) for k, v in env.items()})
        self.p = subprocess.Popen([SIM, "--pty"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, env=e, text=True)
        line = self.p.stderr.readline().strip()
        assert line.startswith("PTY="), line
        self.port = line[4:]

    def close(self):
        self.p.kill()
        self.p.wait()


def tool(sim, *words, extra=()):
    """run tars_serial.main in-process; returns (rc, stdout)"""
    buf = io.StringIO()
    rc = None
    argv = ["--port", sim.port, "--out-dir", OUT] + list(extra) + list(words)
    with contextlib.redirect_stdout(buf):
        try:
            rc = tars_serial.main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 99
            if isinstance(e.code, str):
                buf.write("EXIT: %s\n" % e.code)
    return rc, buf.getvalue()


def newest(kind):
    files = sorted(f for f in os.listdir(OUT) if f.endswith("_%s.tlog" % kind))
    assert files, "no %s recording saved" % kind
    return os.path.join(OUT, files[-1])


def raw_session(sim):
    dev = tars_serial.Device(sim.port)
    assert dev.hello(5) is not None
    dev.send("stop")
    time.sleep(0.2)
    dev.drain()
    return dev


def collect(dev, seconds, until=None):
    out = []
    end = time.time() + seconds
    while time.time() < end:
        line = dev.get(0.1)
        if line:
            out.append(line)
            if until and until(line):
                break
    return out


RESULTS = []


def case(fn):
    def run():
        t0 = time.time()
        try:
            fn()
            RESULTS.append((fn.__name__, True, "%.1fs" % (time.time() - t0)))
            print("  ok   %-32s %.1fs" % (fn.__name__, time.time() - t0))
        except Exception:
            RESULTS.append((fn.__name__, False, traceback.format_exc()))
            print("  FAIL %-32s\n%s" % (fn.__name__, traceback.format_exc()))
    run.__name__ = fn.__name__
    return run


# ---------------------------------------------------------------------------------------------------------------------
S = None   # shared simulator


@case
def test_status_and_help():
    rc, out = tool(S, "status")
    assert rc == 0 and "@ok status" in out and "servo 1" in out, out
    rc, out = tool(S, "help")
    assert "test step" in out and "@ok help" in out


@case
def test_scan():
    rc, out = tool(S, "scan")
    assert out.count("# servo ") == 4 and "NO REPLY" not in out and "7.42 V" in out and "@ok scan" in out, out
    assert "off=0" in out


@case
def test_hold():
    rc, out = tool(S, "test", "hold", "3")
    assert rc == 0 and "== robot still ==" in out, out
    rec = tlog.load(newest("hold"))
    assert rec.kind == "hold" and rec.complete and rec.bad_lines == 0
    rate = len(rec.imu_t) / (rec.imu_t[-1] - rec.imu_t[0])
    assert 85 < rate < 110, rate
    an = tlog.np.linalg.norm(rec.acc.mean(axis=0))
    assert abs(an - 9.78) < 0.1, an
    assert abs(rec.gyro[:, 0].mean() - 0.4) < 0.1
    for i in (1, 2, 3):
        assert len(rec.servo[i]["t"]) > 50, i
    assert len(rec.volt) >= 2 and abs(rec.volt.mean() - 7.42) < 0.01
    assert "gravity points along sensor axis +z" in out


@case
def test_step():
    rc, out = tool(S, "test", "step", "1", "40", "700")
    assert rc == 0 and "== step response ==" in out, out
    rec = tlog.load(newest("step"))
    assert rec.kind == "step" and rec.meta["args"] == "id=1 amp=40 hold_ms=700", rec.meta
    rows = tlog.step_metrics(rec, 1, 40)
    assert len(rows) == 4, rows
    for r in rows:
        assert not r["note"], r
        assert 12 < r["delay_ms"] < 35 and 40 < r["rise_ms"] < 80, r          # sim servo: 6 ms transport, acceleration limited start
        assert r["overshoot_pct"] < 3 and abs(r["final_err"]) < 1.5 and r["vmax"] > 500, r
    assert "barely moved" not in out
    s = rec.servo[1]
    assert abs(s["off"].max() - 40) < 1.5 and abs(s["off"].min() + 40) < 1.5, (s["off"].min(), s["off"].max())
    # servos 2 and 3 stay at zero
    assert abs(rec.servo[2]["off"]).max() < 0.5 and abs(rec.servo[3]["off"]).max() < 0.5
    # focus servo is read faster than the others
    assert len(rec.servo[1]["t"]) > 2.5 * len(rec.servo[2]["t"]), (len(rec.servo[1]["t"]), len(rec.servo[2]["t"]))
    # command log shows the test waveform with the 'test' source tag
    assert (rec.cmd_src == 1).sum() > 100 and rec.cmd_off[:, 0].max() == 40 and rec.cmd_off[:, 0].min() == -40


@case
def test_step_negative_servo3():
    rc, out = tool(S, "test", "step", "3", "-50")
    assert rc == 0, out
    rec = tlog.load(newest("step"))
    s = rec.servo[3]
    # amplitude -50 first: offset = mask * (reg - zero); the response must follow the command in the same units
    assert s["off"].min() < -45 and s["off"].max() > 45
    assert rec.cmd_off[:, 2].min() == -50 and rec.cmd_off[:, 2].max() == 50


@case
def test_sine():
    rc, out = tool(S, "test", "sine", "2", "30", "1.5", "5")
    assert rc == 0 and "gain" in out, out
    rec = tlog.load(newest("sine"))
    lines = tlog.analyze_sine(rec)
    g = [l for l in lines if "gain" in l][0]
    gain = float(g.split("gain = ")[1].split()[0])
    lag = float(g.split("phase lag = ")[1].split()[0])
    assert 0.9 < gain < 1.02 and 8 < lag < 30, g


@case
def test_walk_and_gait_analysis():
    rc, out = tool(S, "test", "walk", "20", "4")
    assert rc == 0 and "== motion ==" in out, out
    rec = tlog.load(newest("walk"))
    assert rec.imu_vx.max() == 20 and rec.kind == "walk"
    f = [l for l in tlog.analyze_gait(rec) if "dominant" in l][0]
    freq = float(f.split(":")[1].split()[0])
    assert 1.15 < freq < 1.35, f
    # servo targets of the classic gait are recorded
    assert abs(rec.cmd_off[:, 0]).max() > 15


@case
def test_turn():
    rc, out = tool(S, "test", "turn", "-15", "2")
    assert rc == 0, out
    rec = tlog.load(newest("turn"))
    assert rec.imu_vyaw.min() == -15


@case
def test_rec_until_stop():
    dev = raw_session(S)
    try:
        dev.send("rec 0")
        lines = collect(dev, 2.0, lambda l: l.startswith("@ok rec started"))
        assert any(l.startswith("@ok rec started") for l in lines), lines
        dev.send("mark pushed here")
        lines = collect(dev, 0.5)
        assert any("@ok mark" in l for l in lines), lines
        time.sleep(0.5)
        dev.send("stop")
        lines = collect(dev, 6.0, lambda l: l.startswith("@ok rec done"))
        assert any(l.startswith("@begin dump kind=rec") for l in lines), lines[:5]
        assert any(l.startswith("@end dump") for l in lines)
        assert lines[-1].startswith("@ok rec done"), lines[-3:]
        rec = tlog.load([l for l in lines if l.startswith("@")])
        assert any("pushed here" in t for _, t in rec.marks), rec.marks
        assert rec.bad_lines == 0 and len(rec.imu_t) > 20
        # data survives: dump again after the session
        dev.send("dump")
        lines = collect(dev, 6.0, lambda l: l.startswith("@ok rec done"))
        assert any(l.startswith("@end dump") for l in lines)
    finally:
        dev.close()


@case
def test_busy_and_stop():
    dev = raw_session(S)
    try:
        dev.send("rec 30")
        collect(dev, 1.0, lambda l: l.startswith("@ok rec started"))
        dev.send("rec 5")
        dev.send("test step 1 20")
        dev.send("watch")
        lines = collect(dev, 1.0)
        assert sum(1 for l in lines if l.startswith("@err") and "busy" in l) == 3, lines
        dev.send("status")
        lines = collect(dev, 1.0, lambda l: l.startswith("@ok status"))
        assert any("mode=rec" in l and "recording=1" in l for l in lines), lines
        dev.send("stop")
        lines = collect(dev, 8.0, lambda l: l.startswith("@ok rec done"))
        assert lines and lines[-1].startswith("@ok rec done"), lines[-3:]
    finally:
        dev.close()


@case
def test_error_paths():
    cases = [
        ("test step 9 40", "usage"), ("test step 1 0", "amplitude"), ("test step 1 500", "amplitude"), ("test step 3 111", "amplitude"),
        ("test sine 1 30 99 3", "frequency"), ("test sine 1 30 0.1 3", "frequency"), ("test walk 99 3", "usage"), ("test walk 0 3", "usage"),
        ("test turn 10 100", "usage"), ("test hold 0", "usage"), ("test hold abc", "usage"), ("test nonsense", "unknown test"), ("test", "usage"),
        ("pose 999 0 0", "limited"), ("pose 1 2", "usage"), ("pose a b c", "not a number"), ("pose 1 2 3 9999", "seconds"),
        ("torque 5", "usage"), ("watch 99", "usage"), ("rec -1", "usage"), ("rec x", "usage"), ("cal 2", "usage"), ("map 2 0", "usage"),
        ("selftest 9", "usage"), ("learn on", "calibrated"), ("learn", "usage"), ("bogus", "unknown command")
    ]
    dev = raw_session(S)
    try:
        for cmd, needle in cases:
            dev.send(cmd)
            lines = collect(dev, 1.5, lambda l: l.startswith("@err") or l.startswith("@ok"))
            hit = [l for l in lines if l.startswith("@err")]
            assert hit and needle in hit[0], (cmd, lines)
    finally:
        dev.close()


@case
def test_pose_release_and_hold():
    dev = raw_session(S)
    try:
        dev.send("pose 10 -10 20 60")
        lines = collect(dev, 1.0, lambda l: l.startswith("@ok pose"))
        assert any("@ok pose 10 -10 20" in l for l in lines), lines
        time.sleep(0.6)
        dev.send("status")
        lines = collect(dev, 1.0, lambda l: l.startswith("@ok status"))
        assert any("servos=pose" in l for l in lines), lines
        dev.send("scan")
        lines = collect(dev, 2.0, lambda l: l.startswith("@ok scan"))
        txt = "\n".join(lines)
        assert "servo 1: pos=440 off=10" in txt and "servo 2: pos=530 off=-10" in txt and "servo 3: pos=472 off=20" in txt, txt
        dev.send("release")
        collect(dev, 0.5, lambda l: l.startswith("@ok release"))
        time.sleep(0.8)
        dev.send("scan")
        lines = collect(dev, 2.0, lambda l: l.startswith("@ok scan"))
        txt = "\n".join(lines)
        assert "servo 1: pos=430 off=0" in txt and "servo 3: pos=492 off=0" in txt, txt
    finally:
        dev.close()


@case
def test_selftest_pose_is_held():
    dev = raw_session(S)
    try:
        dev.send("selftest 1")
        collect(dev, 1.0, lambda l: l.startswith("@ok selftest"))
        time.sleep(1.0)                     # the classic gait would have written zero long ago
        dev.send("scan")
        txt = "\n".join(collect(dev, 2.0, lambda l: l.startswith("@ok scan")))
        assert "servo 1: pos=498 off=68" in txt and "servo 2: pos=608 off=68" in txt, txt
        dev.send("selftest 0")
        collect(dev, 1.0, lambda l: l.startswith("@ok selftest"))
        time.sleep(0.8)
        dev.send("scan")
        txt = "\n".join(collect(dev, 2.0, lambda l: l.startswith("@ok scan")))
        assert "servo 1: pos=430 off=0" in txt, txt
    finally:
        dev.close()


@case
def test_torque_off_and_on():
    dev = raw_session(S)
    try:
        dev.send("torque 0")
        collect(dev, 1.0, lambda l: l.startswith("@ok torque"))
        dev.send("watch 10")
        lines = collect(dev, 3.0)
        offs = []
        for l in lines:
            if l.startswith("# w ") and " off=" in l:
                offs.append([int(x) for x in l.split(" off=")[1].split()[0].split(",")])
        assert len(offs) > 8, lines[:5]
        assert max(o[0] for o in offs) - min(o[0] for o in offs) > 40, offs      # legs are moved "by hand"
        dev.send("stop")
        collect(dev, 1.0, lambda l: l.startswith("@ok stop"))
        dev.send("torque 1")
        collect(dev, 1.0, lambda l: l.startswith("@ok torque"))
        time.sleep(1.2)
        dev.send("scan")
        txt = "\n".join(collect(dev, 2.0, lambda l: l.startswith("@ok scan")))
        assert "servo 1: pos=430 off=0" in txt, txt
    finally:
        dev.close()


@case
def test_watch_format():
    rc, out = tool(S, "watch", "5", extra=("--seconds", "1.5"))
    lines = [l for l in out.splitlines() if l.startswith("t=")]
    assert len(lines) >= 5, out
    assert "acc=" in lines[0] and "|a|=9." in lines[0] and "gyro=" in lines[0] and "off=0,0,0" in lines[-1] and "V=7.42" in lines[-1], lines[-1]


@case
def test_live_stream():
    rc, out = tool(S, "live", extra=("--seconds", "1.5"))
    f = newest("live")
    rec = tlog.load(f)
    assert len(rec.imu_t) > 80 and rec.bad_lines == 0, (len(rec.imu_t), rec.bad_lines)


@case
def test_cal_and_learn_flow():
    dev = raw_session(S)
    try:
        for cmd, needle in [("cal 0", "@ok cal step=0 calibrated=0"), ("cal 1", "@ok cal step=1 calibrated=1"), ("learn dry", "@ok learn"),
                            ("learn on", "@err"), ("learn on go", "@ok learn"), ("learn off", "@ok learn"), ("map 1 0", "@ok map"),
                            ("map 0 0", "@ok map"), ("info", "@ok hello")]:
            dev.send(cmd)
            lines = collect(dev, 2.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
            assert any(needle in l for l in lines), (cmd, lines)
    finally:
        dev.close()


@case
def test_noise_ctrl_c_long_line():
    dev = raw_session(S)
    try:
        dev.ser.write(b"\x00\xff\x1b[A garbage \x7f\x7f\x7f\nstatus\n")
        lines = collect(dev, 1.5, lambda l: l.startswith("@ok status") or l.startswith("@err"))
        assert any(l.startswith("@ok status") for l in lines) or any(l.startswith("@err unknown") for l in lines), lines
        dev.send("status")
        assert any(l.startswith("@ok status") for l in collect(dev, 1.5, lambda l: l.startswith("@ok status")))
        dev.send("watch")
        collect(dev, 0.5)
        dev.ser.write(b"\x03")
        lines = collect(dev, 1.0, lambda l: l.startswith("@ok estop"))
        assert any(l.startswith("@ok estop") for l in lines), lines
        dev.send("x" * 400)
        lines = collect(dev, 1.0, lambda l: l.startswith("@err"))
        assert any("too long" in l for l in lines), lines
    finally:
        dev.close()


@case
def test_checksum_robustness():
    good = [tlog.make_line("I,100,1,2,3,4,5,6,7,8,9,1,-1,0"), tlog.make_line("S,110,1,512,-3,40,900"), tlog.make_line("V,120,742"),
            tlog.make_line("C,130,1,430,540,492,0"), tlog.make_line("M,140,hello, world")]
    lines = ["@begin dump kind=rec events=5 dropped=0", "@meta fw=tars-log/1 zero=430,540,492 mask=1,1,-1 acc_lsb_per_g=16384 g_mm_s2=9800 gyro_lsb_per_dps=32 up_scale=10000", "@meta args=-"]
    lines += good
    lines += [good[0][:-2] + "00", "@I,garbage*12", "@S,1,2,3*00", "boot noise text", good[1].replace("512", "513"), "@Q,1,2*00", ""]
    lines += ["@end dump events=5 skipped=0 dropped=0"]
    rec = tlog.load(lines)
    assert rec.bad_lines == 5 and rec.n_events == 5, (rec.bad_lines, rec.n_events)
    assert len(rec.imu_t) == 1 and len(rec.servo[1]["t"]) == 1 and rec.marks == [(140e-6, "hello, world")]
    assert tlog.decode_speed(-32763) == -5 and tlog.decode_speed(5) == 5 and tlog.decode_load(0x400 | 77) == -77 and tlog.decode_load(77) == 77
    assert tlog.unpack_vxy((20 << 8) | 0xF1) == (20, -15) and tlog.unpack_vxy((-3 << 8) | 7) == (-3, 7)
    tlog.analyze(rec)      # must not raise on a tiny recording


@case
def test_fuzz_commands():
    """random command lines (valid, half valid, garbage) must never wedge or crash the shell"""
    import random
    rnd = random.Random(1234)
    words = ["help", "status", "scan", "watch", "rec", "live", "stop", "dump", "clear", "mark", "test", "pose", "release", "torque", "cal", "learn",
             "map", "selftest", "hello", "step", "sine", "walk", "turn", "hold", "on", "off", "go", "dry"]
    nums = ["0", "1", "2", "3", "-1", "40", "-40", "120", "121", "999999999999", "1.5", "0.2", "abc", "", "+5", "-", "2147483647", "-2147483648"]
    dev = raw_session(S)
    try:
        for i in range(500):
            n = rnd.randint(1, 6)
            line = " ".join(rnd.choice(words) if rnd.random() < 0.5 else rnd.choice(nums) for _ in range(n))
            if rnd.random() < 0.05:
                line = "".join(chr(rnd.randint(1, 255)) for _ in range(rnd.randint(1, 30)))
            if line.split()[:1] in (["reboot"],):
                continue
            try:
                dev.ser.write((line + "\n").encode("latin-1", "replace"))
            except Exception:
                pass
            if i % 25 == 0:
                time.sleep(0.05)
        time.sleep(0.5)
        dev.send("stop")
        time.sleep(0.3)
        dev.send("release")
        time.sleep(0.3)
        dev.drain()
        dev.send("stop")
        collect(dev, 6.0, lambda l: l.startswith("@ok stop"))
        time.sleep(1.0)
        dev.drain()
        dev.send("status")
        lines = collect(dev, 3.0, lambda l: l.startswith("@ok status"))
        assert any(l.startswith("@ok status") for l in lines), lines[-5:]
        assert any("mode=idle" in l for l in lines), lines[:3]
    finally:
        dev.close()
    assert S.p.poll() is None, "simulator crashed"


@case
def test_link_command_and_console_flag():
    dev = raw_session(S)
    try:
        dev.send("link console")
        assert any("@ok link console" in l for l in collect(dev, 1.5, lambda l: l.startswith("@ok") or l.startswith("@err")))
        dev.send("hello")
        assert any("link=console" in l for l in collect(dev, 1.5, lambda l: l.startswith("@ok hello")))
        dev.send("link bogus")
        assert any(l.startswith("@err") for l in collect(dev, 1.5, lambda l: l.startswith("@err")))
        dev.send("link usb")
        assert any("@ok link usb" in l for l in collect(dev, 1.5, lambda l: l.startswith("@ok")))
    finally:
        dev.close()
    rc, out = tool(S, "scan", extra=("--console",))
    assert rc == 0 and "@ok scan" in out, out
    tool(S, "link", "usb")


@case
def test_tool_ctrl_c_during_recording_and_watch():
    """Ctrl-C in the PC tool: the robot stops walking, the recording that was running is still saved"""
    import signal
    tool_py = os.path.join(HERE, "..", "tars_serial.py")
    out_dir = tempfile.mkdtemp(prefix="tars_ctrlc_")
    p = subprocess.Popen([sys.executable, tool_py, "--port", S.port, "--out-dir", out_dir, "test", "walk", "20", "50"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(3.0)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0 and "saved" in out and "complete=True" in out, out[-800:]
    files = [f for f in os.listdir(out_dir) if f.endswith("_walk.tlog")]
    assert len(files) == 1
    rec = tlog.load(os.path.join(out_dir, files[0]))
    assert rec.imu_vx.max() == 20 and rec.imu_t[-1] < 49, "walk should have been stopped early"
    # the robot is idle again
    rc, out = tool(S, "status")
    assert "mode=idle" in out, out
    # same for watch
    p = subprocess.Popen([sys.executable, tool_py, "--port", S.port, "--out-dir", out_dir, "watch"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=30)
    assert out.count("t=") > 5, out[-500:]
    rc, out = tool(S, "status")
    assert "mode=idle" in out, out


@case
def test_fast_baud_negotiation_and_restore():
    rc, out = tool(S, "scan", extra=("--fast", "--fast-baud", "230400"))
    assert rc == 0 and "fast link: 230400 baud" in out and "@ok scan" in out, out
    rc, out = tool(S, "hello")
    assert "baud=115200" in out, out          # the tool put the robot's UART back on exit
    dev = raw_session(S)
    try:
        for cmd, expect in (("baud 12345", "@err"), ("baud abc", "@err"), ("baud", "@err"), ("baud 921600", "@ok baud 921600"), ("baud ok", "@ok baud confirmed"),
                            ("baud 115200", "@ok baud 115200")):
            dev.send(cmd)
            lines = collect(dev, 1.5, lambda l: l.startswith("@ok") or l.startswith("@err"))
            assert any(l.startswith(expect) for l in lines), (cmd, lines)
    finally:
        dev.close()


@case
def test_imu_diagnostics():
    dev = raw_session(S)
    try:
        def run(cmd, until, t=3.0):
            dev.send(cmd)
            return collect(dev, t, until)
        done = lambda l: l.startswith("@ok") or l.startswith("@err")
        lines = run("imu", done)
        txt = "\n".join(lines)
        assert "# imu 0x00..0x0a 05 7c 40 04 64 00 00 00 03 00 00" in txt, txt
        assert "# imu raw acc=1140,16335,1190 gyro=7,171,-8 temp=25.5C" in txt and lines[-1] == "@ok imu regs", txt
        lines = run("imu raw 3", done, 4.0)
        assert sum(1 for l in lines if l.startswith("# imu raw")) == 3 and lines[-1] == "@ok imu raw", lines
        lines = run("imu fs 1", done)
        assert any("# imu 0x00..0x0a 05 7c 40 14 64" in l for l in lines), lines        # CTRL2 = 0x14 -> +-4 g
        lines = run("imu reinit", done)
        txt = "\n".join(lines)
        assert "RST_RESULT=0x80" in txt and "# imu 0x00..0x0a 05 7c 40 04 64" in txt, txt  # back to the stock configuration
        for cmd in ("imu fs 9", "imu fs", "imu raw 0", "imu raw x", "imu bogus"):
            lines = run(cmd, done)
            assert lines and lines[-1].startswith("@err"), (cmd, lines)
    finally:
        dev.close()


@case
def test_estop_dead_man_timer_and_mapping_forms():
    dev = raw_session(S)
    try:
        def run(cmd, until, t=3.0):
            dev.send(cmd)
            return collect(dev, t, until)
        done = lambda l: l.startswith("@ok") or l.startswith("@err")
        assert any("@ok cal step=1 calibrated=1" in l for l in run("cal 1", done))           # the simulator is calibrated from here on
        # dead-man timer: 'learn on go 2' switches itself off after 2 s (simulator time runs 6x)
        lines = run("learn on go 2", done)
        assert lines[-1].startswith("@ok learn") and "learned=on" in lines[-1], lines
        lines = run("status", lambda l: l.startswith("@ok status"))
        assert any("learned=on" in l for l in lines), lines
        lines = collect(dev, 3.0, lambda l: "switched itself off" in l)
        assert any("switched itself off" in l for l in lines), lines
        lines = run("status", lambda l: l.startswith("@ok status"))
        assert any("learned=off" in l for l in lines), lines
        # estop switches the learned mode off immediately and stops a running recording session
        assert run("learn on go 600", done)[-1].startswith("@ok learn")
        dev.send("rec 0")
        collect(dev, 1.0, lambda l: l.startswith("@ok rec started"))
        lines = run("estop", lambda l: l.startswith("@ok rec done") or l.startswith("@err"), 6.0)
        assert any(l.startswith("@ok estop") for l in lines) and any("@end dump" in l for l in lines), lines
        lines = run("status", lambda l: l.startswith("@ok status"))
        assert any("learned=off" in l for l in lines) and any("mode=idle" in l for l in lines), lines
        # argument checks
        for cmd in ("learn on go 0", "learn on go 601", "learn on go abc", "learn on", "learn bogus"):
            lines = run(cmd, done)
            assert lines[-1].startswith("@err"), (cmd, lines)
        # servo sign mapping: both forms
        for cmd, needle in (("map +1 -1 +1 0", "map=+-+"), ("map -1 -1 1 1", "map=--+"), ("map 1 1 1 0", "map=+++"), ("map 1 0", "map=---"), ("map 0 0", "map=+++")):
            lines = run(cmd, done)
            assert lines[-1].startswith("@ok map") and needle in lines[-1], (cmd, lines)
        for cmd in ("map 2 0", "map 1 1 1", "map 1 1 1 2", "map 0 1 1 0", "map a b c d", "map"):
            lines = run(cmd, done)
            assert lines[-1].startswith("@err"), (cmd, lines)
    finally:
        dev.close()


@case
def test_learn_auto_and_log():
    dev = raw_session(S)
    try:
        def run(cmd, until=None, t=3.0):
            dev.send(cmd)
            return collect(dev, t, until or (lambda l: l.startswith("@ok") or l.startswith("@err")))
        assert run("cal 1")[-1].startswith("@ok")                    # the shared simulator is calibrated from here on
        for cmd, needle in (("learn auto on", "@ok learn auto=1"), ("learn auto walk go", "@ok learn auto=2"), ("learn auto off", "@ok learn auto=0")):
            lines = run(cmd)
            assert needle in lines[-1] and "next power-up" in lines[-1], (cmd, lines)
        for cmd in ("learn auto walk", "learn auto", "learn auto maybe"):
            assert run(cmd)[-1].startswith("@err"), cmd
        lines = run("learn log")
        assert sum(1 for l in lines if l.startswith("# hist ")) == 3 and "@ok learn log 3 entries" in lines[-1], lines
        assert any("FALLEN" in l for l in lines) and any("GETUP1" in l for l in lines), lines
        helptext = "\n".join(run("help", lambda l: l.startswith("@ok"), 3.0))
        assert "learn auto off|on|walk go" in helptext and "learn log" in helptext and "learn tune" in helptext and "learn gait" in helptext, helptext
    finally:
        dev.close()


@case
def test_learn_tune():
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert "@ok learn tune kp=100% kd=100% lpf=0Hz amp=100% hz=500" in run("learn tune")[-1], "500 Hz is the default"
        assert "kp=100% kd=40% lpf=4Hz amp=90% hz=500" in run("learn tune 100 40 4 90")[-1]
        assert "kp=100% kd=0% lpf=4Hz amp=90% hz=500" in run("learn tune 100 0")[-1]               # fewer arguments keep the rest
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=500" in run("learn tune 60")[-1]
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=100" in run("learn tune 60 0 4 90 100")[-1]
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=500" in run("learn tune 60 0 4 90 500")[-1]
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=250" in run("learn tune 60 0 4 90 250")[-1]
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=100" in run("learn tune 60 0 4 90 100")[-1]
        for bad in ("learn tune 301", "learn tune 100 100 51", "learn tune 100 100 0 10", "learn tune x", "learn tune 1 2 3 4 50 6", "learn tune -1",
                    "learn tune 100 100 0 100 75", "learn tune 100 100 0 100 25", "learn tune 100 100 0 100 400", "learn tune 100 100 0 100 501"):
            assert run(bad)[-1].startswith("@err"), bad
        assert "kp=60% kd=0% lpf=4Hz amp=90% hz=100" in run("learn tune")[-1]                      # refused values change nothing
        assert "kp=100% kd=100% lpf=0Hz amp=100% hz=50" in run("learn tune 100 100 0 100 50")[-1]
    finally:
        dev.close()


@case
def test_learn_gait_set():
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert "@ok learn gait=6 (v9, sinusoid hips)" in run("learn gait")[-1], "the newest set is the default"
        assert "@ok learn gait=1 (v5)" in run("learn gait 1")[-1]
        assert "@ok learn gait=2 (v5, faster forward)" in run("learn gait 2")[-1]
        assert "@ok learn gait=3 (v6, foot plates)" in run("learn gait 3")[-1]
        assert "@ok learn gait=4 (v7, smoother)" in run("learn gait 4")[-1]
        assert "@ok learn gait=5 (v8, smooth fwd+back)" in run("learn gait 5")[-1]
        assert "@ok learn gait=6 (v9, sinusoid hips)" in run("learn gait 6")[-1]
        assert "@ok learn gait=0 (v4)" in run("learn gait 0 save")[-1]
        for bad in ("learn gait 7", "learn gait x", "learn gait 1 keep", "learn gait 1 save now", "learn gait -1"):
            assert run(bad)[-1].startswith("@err"), bad
        assert "@ok learn gait=0 (v4)" in run("learn gait")[-1]
    finally:
        dev.close()


@case
def test_learn_gait_custom():
    """`learn gait custom <f|b|t|r> <15 integers>` replaces one gait of the current set (volatile); bad input is refused; `learn gait N` brings the set back"""
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        good = "41623 -48900 -141548 -13390 -29472 4023 -3409 1239 -2122 -665 1568486 -102677 -12904 2267 -1215"
        assert "custom" not in run("learn gait 1")[-1]
        assert "@ok learn gait=1 (v5) custom=b" in run("learn gait custom b " + good)[-1]
        assert "@ok learn gait=1 (v5) custom=fb" in run("learn gait custom f " + good)[-1]
        assert "custom=fb" in run("learn gait")[-1]
        for bad in ("learn gait custom", "learn gait custom b", "learn gait custom b " + good + " 7", "learn gait custom b " + " ".join(good.split()[:14]),
                    "learn gait custom x " + good, "learn gait custom bb " + good, "learn gait custom b " + good.replace("41623", "4.1"),
                    "learn gait custom b " + good.replace("1568486", "100000"), "learn gait custom b " + good.replace("1568486", "9000000")):
            r = run(bad)[-1]
            assert r.startswith("@err"), (bad, r)
        assert "custom=fb" in run("learn gait")[-1], "failed commands must not change anything"
        assert run("learn gait 4")[-1] == "@ok learn gait=4 (v7, smoother)", run("learn gait")[-1]
        assert "custom" not in run("learn gait")[-1]
    finally:
        dev.close()


@case
def test_learn_speed():
    """`learn speed [on|off]`: the walking command's magnitude scales the gait amplitude (default on)"""
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert run("learn speed")[-1] == "@ok learn speed=on max=90"
        assert run("learn speed off")[-1] == "@ok learn speed=off max=90"
        assert run("learn speed on")[-1] == "@ok learn speed=on max=90"
        assert run("learn speed max 100")[-1] == "@ok learn speed=on max=100"
        assert run("learn speed max 80")[-1] == "@ok learn speed=on max=80"
        for bad in ("learn speed maybe", "learn speed on 50", "learn speed 1", "learn speed max", "learn speed max 60", "learn speed max 101", "learn speed max x", "learn speed max 90 1"):
            assert run(bad)[-1].startswith("@err"), bad
    finally:
        dev.close()


@case
def test_learn_getup():
    """`learn getup [kf|alt|nn] [save]`: which get-up the robot uses after a fall; bad input is refused"""
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert run("learn getup")[-1] == "@ok learn getup=alt network=compiled in", "the network first, keyframes on the 2nd attempt, is the default"
        assert run("learn getup nn")[-1] == "@ok learn getup=nn network=compiled in"
        assert run("learn getup kf save")[-1] == "@ok learn getup=kf network=compiled in"
        assert run("learn getup alt")[-1] == "@ok learn getup=alt network=compiled in"
        for bad in ("learn getup maybe", "learn getup nn keep", "learn getup nn save now", "learn getup 1", "learn getup kf save 1"):
            assert run(bad)[-1].startswith("@err"), bad
        assert run("learn getup")[-1] == "@ok learn getup=alt network=compiled in", "failed commands must not change anything"
    finally:
        dev.close()


@case
def test_learn_straight():
    """`learn straight [on|off|kp kd [trim]]`: heading hold of the forward / backward walks (volatile); numbers switch it on, bad input is refused"""
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert run("learn straight")[-1] == "@ok learn straight=on kp=150% kd=0% trim=0 permille", "on at power-up"
        assert run("learn straight off")[-1] == "@ok learn straight=off kp=150% kd=0% trim=0 permille"
        assert run("learn straight on")[-1] == "@ok learn straight=on kp=150% kd=0% trim=0 permille"
        assert run("learn straight 80 5")[-1] == "@ok learn straight=on kp=80% kd=5% trim=0 permille", "numbers switch it on"
        assert run("learn straight 80 5 -20")[-1] == "@ok learn straight=on kp=80% kd=5% trim=-20 permille"
        assert run("learn straight off")[-1] == "@ok learn straight=off kp=80% kd=5% trim=-20 permille", "off keeps the numbers"
        assert run("learn straight on")[-1] == "@ok learn straight=on kp=80% kd=5% trim=-20 permille"
        assert run("learn straight 0 0 25")[-1] == "@ok learn straight=on kp=0% kd=0% trim=25 permille", "a pure trim"
        for bad in ("learn straight maybe", "learn straight on 5", "learn straight 50", "learn straight 301 0", "learn straight 50 101", "learn straight -1 0", "learn straight 50 0 301",
                    "learn straight 50 0 -301", "learn straight x 0", "learn straight 50 0 1 2"):
            assert run(bad)[-1].startswith("@err"), bad
        assert run("learn straight")[-1] == "@ok learn straight=on kp=0% kd=0% trim=25 permille", "failed commands must not change anything"
    finally:
        dev.close()


@case
def test_learn_motion():
    """`learn motion [<name> [speed 30..100] | stop]`: lists the show motions; needs the learned mode ON; one at a time; bad names / speeds / extra words are refused"""
    dev = raw_session(S)
    try:
        def run(cmd):
            dev.send(cmd)
            return collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert run("cal 1")[-1].startswith("@ok")                                   # the shared simulator is calibrated from here on
        assert run("learn off")[-1].startswith("@ok")
        assert run("learn motion")[-1] == ("@ok learn motion: nod bow headbang happy sad angry scared surprised shy curious sleepy proud laugh loving impatient wave cheer windmill robotdance swim twist   "
                                           "(learn motion <name> [speed 30..100] | learn motion stop)")
        r = run("learn motion bow")[-1]
        assert r.startswith("@err") and "learned mode is off" in r, r
        assert run("learn on go 600")[-1].startswith("@ok learn")
        r = run("learn motion bow")[-1]
        assert r.startswith("@ok motion bow accepted, speed 100%") and "learn motion stop" in r, r
        r = run("learn motion nod")[-1]
        assert r.startswith("@err") and "busy" in r, r                              # one at a time
        r = run("learn motion stop")[-1]
        assert r.startswith("@ok motion stop requested"), r
        assert run("learn motion headbang 50")[-1].startswith("@ok motion headbang accepted, speed 50%")
        assert run("learn motion stop")[-1].startswith("@ok motion stop requested")
        for bad in ("learn motion dance", "learn motion bow 29", "learn motion bow 101", "learn motion bow x", "learn motion bow 50 1", "learn motion stop now", "learn motion headbang -5", "learn motion sneeze"):
            r = run(bad)[-1]
            assert r.startswith("@err"), (bad, r)
        assert "unknown motion 'dance'" in run("learn motion dance")[-1] and "headbang happy sad angry" in run("learn motion dance")[-1]
        assert run("learn motion headbang 30")[-1].startswith("@ok motion headbang accepted, speed 30%"), "the lowest speed is accepted"
        assert run("learn motion stop")[-1].startswith("@ok")
        assert run("learn motion twist 70")[-1].startswith("@ok motion twist accepted, speed 70%"), "the last motion of the list"
        assert run("learn motion stop")[-1].startswith("@ok")
        dev.send("help")
        helptext = "\n".join(collect(dev, 3.0, lambda l: l.startswith("@ok")))
        assert "learn motion [<name> [speed%]|stop]" in helptext, helptext
        assert run("learn off")[-1].startswith("@ok learn")
    finally:
        dev.close()


@case
def test_plain_walk_with_watch_and_then():
    """'walk' / 'turn' without a recording, started by --then after `watch` is running (the live status line shows the walk), and the usage checks"""
    dev = raw_session(S)
    try:
        def run(cmd, t=3.0):
            dev.send(cmd)
            return collect(dev, t, lambda l: l.startswith("@ok") or l.startswith("@err"))
        for bad in ("walk", "walk 0 5", "walk 99 5", "walk 20 0", "walk 20 61", "walk x 5", "turn 99 5", "turn 20"):
            assert run(bad)[-1].startswith("@err"), bad
        assert run("walk 20 1")[-1].startswith("@ok walk started")
        assert run("turn -10 1")[-1].startswith("@ok turn started")
        run("stop")
    finally:
        dev.close()
    rc, out = tool(S, "watch", extra=["--seconds", "3", "--then", "walk 20 2"])
    assert rc == 0 and "then: walk 20 2" in out and "@ok walk started" in out and " acc=" in out and " st=" in out, (rc, out[:400])
    rc, out = tool(S, "status", extra=["--then", "walk 20 2"])
    assert rc != 0 and "--then goes with" in out, (rc, out)


@case
def test_walk_with_delay_starts_later():
    """'test walk <vx> <sec> <delay>': the recording starts at once, the robot moves after the delay (unplug the USB cable meanwhile)"""
    dev = raw_session(S)
    try:
        for bad in ("test walk 20 2 -1", "test walk 20 2 121", "test walk 20 2 x", "test walk 20 2 3 4"):
            dev.send(bad)
            assert collect(dev, 2.0, lambda l: l.startswith("@ok") or l.startswith("@err"))[-1].startswith("@err"), bad
        dev.send("test walk 20 2 3")
        lines = collect(dev, 2.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
        assert lines[-1].startswith("@ok walk will start in 3 s"), lines
        lines = collect(dev, 20.0, lambda l: l.startswith("@ok walk done"))
        assert lines and lines[-1].startswith("@ok walk done"), lines[-3:]
        rec = tlog.load([l for l in lines if l.startswith("@")])
        vx = [int(v) for v in rec.imu_vx]
        first_walk = next((i for i, v in enumerate(vx) if v != 0), None)
        assert first_walk is not None and first_walk > 100, "the walk must start ~3 s (>100 samples) after the recording: %s" % first_walk
        assert vx[0] == 0 and vx[-1] == 0 and 20 in vx, "idle first, walking in the middle, stopped at the end"
        assert abs(float(rec.imu_t[first_walk] - rec.imu_t[0]) - 3.0) < 0.6, rec.imu_t[first_walk] - rec.imu_t[0]
        # 'stop' before the delay is over cancels the walk
        dev.send("test walk 20 2 5")
        collect(dev, 2.0, lambda l: l.startswith("@ok"))
        dev.send("stop")
        collect(dev, 6.0, lambda l: l.startswith("@ok rec done"))
        time.sleep(1.5)
        dev.send("status")
        st = collect(dev, 2.0, lambda l: l.startswith("@ok status"))
        assert any("mode=idle" in l for l in st), st
    finally:
        dev.close()


def test_learn_auto_needs_the_imu_calibration():
    sim = Sim()
    try:
        dev = raw_session(sim)
        for cmd, ok in (("learn auto on", False), ("learn auto walk go", False), ("learn auto off", True)):
            dev.send(cmd)
            lines = collect(dev, 3.0, lambda l: l.startswith("@ok") or l.startswith("@err"))
            assert lines[-1].startswith("@ok" if ok else "@err"), (cmd, lines)
        dev.close()
    finally:
        sim.close()


@case
def test_check_command():
    rc, out = tool(S, "check")
    assert rc == 0 and out.count("[ ok ]") >= 7 and "[FAIL]" not in out, out
    assert "gravity points along sensor axis +z" in out and "servo bus voltage 7.42 V" in out, out


@case
def test_learned_timeline_analysis():
    lines = ["@begin dump kind=walk events=0 dropped=0", "@meta fw=tars-log/1 zero=430,540,492 mask=1,1,-1 acc_lsb_per_g=16384 g_mm_s2=9800 gyro_lsb_per_dps=32 up_scale=10000 vxy=int8x2",
             "@meta args=vx=20 sec=10"]
    # 10 s at 100 Hz: STAND 0-1 s, WALK_F 1-5 s, FALLEN 5-6.5 s, GETUP1 6.5-8 s, GETUP2C 8-8.5, STAND afterwards; flags: filter ready + learned on
    def state_at(t):
        return 0 if t < 1 else 1 if t < 5 else 5 if t < 6.5 else 6 if t < 8 else 9 if t < 8.5 else 0
    for i in range(1000):
        t = i * 0.01
        st = state_at(t)
        upz = 0.2 if st in (5, 6) else 0.99
        lines.append(tlog.make_line("I,%d,0,16384,0,0,0,0,%d,0,%d,5,%d,0" % (int(t * 1e6), int(-0.1 * 10000), int(upz * 10000), st)))
    lines.append("@end dump events=1000 skipped=0 dropped=0")
    rec = tlog.load(lines)
    out = "\n".join(tlog.analyze_learned(rec))
    assert "falls detected: 1   get-up attempts: 1   FAILED: 0" in out, out
    assert "WALK_F" in out and "fall at 5.0 s -> STAND again at 8.5 s" in out, out
    assert "lowest up_z 0.20" in out, out
    assert "== learned behaviour ==" in "\n".join(tlog.analyze(rec))


@case
def test_html_and_csv():
    rec = tlog.load(newest("step"))
    html_text = tlog.report_html(rec, "t")
    assert html_text.count("<svg") >= 7 and "servo 1 position offset" in html_text
    f = io.StringIO()
    tlog.to_csv(rec, f)
    rows = f.getvalue().splitlines()
    assert len(rows) == len(rec.imu_t) + 1 and rows[0].startswith("t_s,acc_x")


def test_ring_full():
    sim = Sim(SIM_RING=300)
    try:
        rc, out = tool(sim, "rec", "4")
        assert "buffer full" in out and "== summary ==" in out, out
        rec = tlog.load(newest("rec"))
        assert rec.n_events == 300 and int(rec.begin["dropped"]) >= 0, (rec.n_events, rec.begin)
        assert rec.complete
    finally:
        sim.close()


def test_dead_servo_and_calibration_mode():
    sim = Sim(SIM_DEAD=2)
    try:
        rc, out = tool(sim, "scan")
        assert "servo 2: NO REPLY" in out and "servo 1: pos=430" in out, out
    finally:
        sim.close()
    sim = Sim(SIM_CALIB=1)
    try:
        for words in (("test", "step", "1", "40"), ("pose", "1", "1", "1"), ("torque", "0"), ("selftest", "1"), ("test", "walk", "10", "2")):
            rc, out = tool(sim, *words)
            assert "calibration mode" in out, (words, out)
    finally:
        sim.close()


def test_stuck_accelerometer_axis_is_reported():
    sim = Sim(SIM_STUCK_Z=1)
    try:
        rc, out = tool(sim, "check")
        assert rc == 1, out
        assert "[FAIL] accelerometer z axis is stuck at the full-scale code" in out, out
        assert "[warn] |acceleration|" in out and "stuck axis ignored" in out, out
        rec = tlog.load(newest("hold"))
        assert (rec.imu_flags & 32).all()
        assert any("WARNING: accelerometer axis z is stuck" in l for l in tlog.analyze_still(rec))
    finally:
        sim.close()


class ScriptIO:
    """answers the prompts of setup_flow like a person would; the posture prompts steer the simulated robot through a control file"""

    def __init__(self, ctl_path, choices, confirm=True):
        self.ctl, self.choices, self.confirm_answer = ctl_path, list(choices), confirm
        self.log = []
        self._post("0.0 0.999")

    def _post(self, text):
        with open(self.ctl, "w") as f:
            f.write(text)
        time.sleep(0.12)                       # the simulator re-reads the file every 50 ms (virtual)

    def say(self, text=""):
        self.log.append(text)

    def pause(self, text):
        self.log.append("PAUSE: " + text)
        if "仰面" in text:
            self._post("0.0 -1.0")             # lying on its back
        elif "向前倾" in text or "倾斜约 30" in text:
            self._post(getattr(self, "tilt", "-0.5 0.866"))
        else:
            self._post("0.0 0.999")            # upright

    def choose(self, text, options):
        self.log.append("ASK: " + text)
        assert self.choices, "unexpected question: " + text
        return self.choices.pop(0)

    def confirm(self, text):
        self.log.append("CONFIRM: " + text)
        return self.confirm_answer


def run_setup(choices, env=None, tilt=None, confirm=True, stages=("pre", "imu", "map", "dry")):
    import setup_flow
    setup_flow.TIME_SCALE = 1.0 / 6.0
    ctl = os.path.join(OUT, "ctl_%d.txt" % int(time.time() * 1000))
    sim = Sim(SIM_CTL_FILE=ctl, **(env or {}))
    dev = None
    try:
        dev = tars_serial.Device(sim.port)
        assert dev.hello(5) is not None
        dev.send("stop")
        time.sleep(0.2)
        dev.drain()
        io_ = ScriptIO(ctl, choices, confirm)
        if tilt:
            io_.tilt = tilt
        rc = setup_flow.Setup(dev, io_, verbose=False).run(stages)
        dev.send("status")
        status = collect(dev, 2.0, lambda l: l.startswith("@ok status"))
        return rc, io_.log, status
    finally:
        if dev:
            dev.close()
        sim.close()


@case
def test_setup_flow_default_mapping():
    # ID1: robot-left (viewer right), foot toward the viewer (forward); ID2: other side, foot away (backward); legs retract; both self-tests pass
    rc, log, status = run_setup([2, 1, 1, 2, 1, 1, 1])
    txt = "\n".join(log)
    assert rc == 0, txt
    assert any("map=+++ learned=off cal=1" in l for l in status), status
    assert "标定已保存" in txt and "映射已保存" in txt and "摔倒检测正常" in txt and "✓ 准备好了" in txt, txt


@case
def test_setup_flow_flipped_and_swapped_mapping():
    # ID1 moved the viewer's LEFT leg (= robot right arm -> swap), foot backward; ID2 foot forward; the slider extends the legs
    rc, log, status = run_setup([1, 2, 2, 1, 2, 1, 1])
    assert rc == 0, "\n".join(log)
    assert any("map=---+swapLR" in l for l in status), status


@case
def test_setup_flow_single_servo_flipped():
    rc, log, status = run_setup([2, 1, 1, 1, 1, 1, 1])     # only ID2 (+) moves the foot forward instead of backward
    assert rc == 0, "\n".join(log)
    assert any("map=+-+" in l and "swapLR" not in l for l in status), status


@case
def test_setup_flow_stops_on_inconsistent_observations():
    rc, log, status = run_setup([2, 1, 2, 2, 1, 1, 1])     # both servos "moved the leg on the right": impossible
    txt = "\n".join(log)
    assert rc == 1 and "同一边的腿" in txt, txt
    assert any("map=+++" in l for l in status)               # nothing stored


@case
def test_setup_flow_selftest_disagrees():
    rc, log, status = run_setup([2, 1, 1, 2, 1, 2, 1])     # the person says the self-test feet did NOT swing forward
    assert rc == 1 and "自检不对" in "\n".join(log), "\n".join(log)


@case
def test_setup_flow_preflight_failures():
    rc, log, _ = run_setup([], env={"SIM_DEAD": 2})
    assert rc == 1 and "舵机没有应答" in "\n".join(log)
    rc, log, _ = run_setup([], env={"SIM_STUCK_Z": 1})
    assert rc == 1 and "卡死在满量程" in "\n".join(log), "\n".join(log)


@case
def test_setup_flow_detects_a_backward_tilt():
    rc, log, _ = run_setup([], tilt="0.5 0.866", stages=("imu",))     # the person tilted the robot backwards
    assert rc == 1 and "向后倾" in "\n".join(log), "\n".join(log)


def test_dump_survives_stall():
    """the host stops reading for 1.5 s in the middle of a dump: nothing may be lost, repeated or damaged"""
    sim = Sim(SIM_OUT_FAIL_AFTER=20001, SIM_OUT_STALL_MS=1500)
    try:
        rc, out = tool(sim, "rec", "3")
        assert "cut short" not in out
        rec = tlog.load(newest("rec"))
        assert rec.complete and rec.bad_lines == 0, (rec.complete, rec.bad_lines)
        assert int(rec.end["events"]) == rec.n_events == int(rec.begin["events"]), (rec.end, rec.n_events, rec.begin)
        assert (tlog.np.diff(rec.imu_t) > 0).all(), "IMU samples repeated or out of order"
    finally:
        sim.close()


def test_dump_cut_short():
    """the host is silent for longer than the device waits: the dump is abandoned, the device stays usable, 'dump' sends it again"""
    sim = Sim(SIM_OUT_FAIL_AFTER=20001, SIM_OUT_STALL_MS=12000)
    try:
        dev = raw_session(sim)
        dev.send("rec 3")
        lines = collect(dev, 8.0, lambda l: "cut short" in l)
        assert any("cut short" in l for l in lines), "no notice about the abandoned dump"
        assert not any(l.startswith("@end dump") for l in lines)
        dev.send("status")
        lines = collect(dev, 2.0, lambda l: l.startswith("@ok status"))
        assert any(l.startswith("@ok status") for l in lines), "device must stay responsive after a cut-short dump"
        dev.send("dump")
        lines = collect(dev, 8.0, lambda l: l.startswith("@ok rec done"))
        assert any(l.startswith("@end dump") for l in lines) and lines[-1].startswith("@ok rec done"), lines[-3:]
        rec = tlog.load([l for l in lines if l.startswith("@")])
        assert rec.complete and rec.bad_lines == 0 and int(rec.end["events"]) == rec.n_events
        dev.close()
    finally:
        sim.close()


def test_then_option_starts_a_command_after_the_recording():
    """--then: the recording is running first, the command (learned mode) starts right after; a refused command ends the recording early"""
    sim = Sim()
    saved = lambda out: (re.search(r"saved (\S+\.tlog)", out) or [None, None])[1]
    try:
        for c in ("cal 0", "cal 1"):
            rc, out = tool(sim, c)
            assert rc == 0, out
        rc, out = tool(sim, "rec", "2", extra=["--then", "learn dry"])
        assert rc == 0, out
        assert "then: learn dry" in out and "@ok learn" in out, out
        assert saved(out) and os.path.exists(saved(out)), out
        rec = tlog.load(saved(out))
        assert rec.imu_state[0] < 0 <= rec.imu_state[-1], (rec.imu_state[0], rec.imu_state[-1])      # recording first, learned mode on afterwards
        # 'learn on' without 'go' is refused: the recording is ended (and still saved), exit status 1
        rc, out = tool(sim, "rec", "60", extra=["--then", "learn on"])
        assert rc == 1 and "a --then command failed" in out, (rc, out)
        assert saved(out) and os.path.exists(saved(out)), out
        # it only goes with recording commands
        rc, out = tool(sim, "status", extra=["--then", "learn dry"])
        assert rc != 0 and "--then goes with" in out, (rc, out)
    finally:
        sim.close()


def main():
    global S
    if not os.path.exists(SIM):
        raise SystemExit("build the simulator first: make -C %s" % HERE)
    print("recordings of this run go to", OUT)
    S = Sim()
    try:
        for fn in (test_status_and_help, test_scan, test_hold, test_step, test_step_negative_servo3, test_sine, test_walk_and_gait_analysis, test_turn,
                   test_rec_until_stop, test_busy_and_stop, test_error_paths, test_pose_release_and_hold, test_selftest_pose_is_held,
                   test_torque_off_and_on, test_watch_format, test_live_stream, test_cal_and_learn_flow, test_noise_ctrl_c_long_line,
                   test_checksum_robustness, test_fuzz_commands, test_link_command_and_console_flag, test_fast_baud_negotiation_and_restore, test_tool_ctrl_c_during_recording_and_watch, test_imu_diagnostics, test_estop_dead_man_timer_and_mapping_forms, test_learn_auto_and_log, test_learn_tune, test_learn_gait_set, test_learn_gait_custom, test_learn_speed, test_learn_getup, test_learn_straight, test_learn_motion, test_plain_walk_with_watch_and_then, test_walk_with_delay_starts_later, test_check_command, test_learned_timeline_analysis, test_html_and_csv):
            fn()
    finally:
        S.close()
    for fn in (test_setup_flow_default_mapping, test_setup_flow_flipped_and_swapped_mapping, test_setup_flow_single_servo_flipped,
               test_setup_flow_stops_on_inconsistent_observations, test_setup_flow_selftest_disagrees, test_setup_flow_preflight_failures,
               test_setup_flow_detects_a_backward_tilt):
        fn()                                    # already wrapped by @case; each one spawns its own simulator
    for fn in (test_ring_full, test_dead_servo_and_calibration_mode, test_stuck_accelerometer_axis_is_reported, test_dump_survives_stall, test_dump_cut_short,
               test_then_option_starts_a_command_after_the_recording, test_learn_auto_needs_the_imu_calibration):
        case(fn)()
    failed = [r for r in RESULTS if not r[1]]
    print("\n%d tests, %d failed" % (len(RESULTS), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
