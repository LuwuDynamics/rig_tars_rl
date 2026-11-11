#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tlog.py - parse, analyse and plot recordings of the Tars serial log tool.

A recording ("*.tlog") is the text the robot sends after `rec` / `test ...`:

    @begin dump kind=step events=1234 dropped=0
    @meta fw=tars-log/1 zero=430,540,492 mask=1,1,-1 acc_lsb_per_g=16384 ...
    @I,<t_us>,ax,ay,az,gx,gy,gz,ux,uy,uz,flags,state,vxy*HH      IMU (raw LSB, see @meta)
    @S,<t_us>,<servo id>,pos,spd,load,lat_us*HH                  servo read-back (raw servo registers)
    @C,<t_us>,<src>,p1,p2,p3,speed*HH                            servo targets that were sent (raw servo registers)
    @V,<t_us>,<centivolt>*HH                                     servo bus voltage
    @M,<t_us>,<text>*HH                                          marker
    @end dump ...

Units after decoding: seconds, m/s^2, deg/s, up-vector components (unit length), servo counts.
"Offset" values (`off`) are in WritePos units: mask * (register - zero), i.e. the same numbers as `pose` / `test step`.
Needs numpy for the analysis functions (the capture tool itself does not).
"""
import html
import math
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

DEG_PER_COUNT = 300.0 / 1024.0       # servo travel per count


def need_numpy():
    if np is None:
        raise SystemExit("numpy is required for the analysis (pip install numpy, or use the ESP-IDF python environment)")


# ---------------------------------------------------------------------------------------------------------------------
# line level
# ---------------------------------------------------------------------------------------------------------------------
def line_checksum(text):
    c = 0
    for b in text.encode("latin-1", "replace"):
        c ^= b
    return c


def check_line(line):
    """True if `line` is '@....*HH' with a matching XOR checksum."""
    if not line.startswith("@"):
        return False
    i = line.rfind("*")
    if i < 1 or len(line.rstrip()) != i + 3:
        return False
    try:
        want = int(line[i + 1:i + 3], 16)
    except ValueError:
        return False
    return line_checksum(line[1:i]) == want


def make_line(body):
    """Build a checksummed data line (used by the tests)."""
    return "@%s*%02X" % (body, line_checksum(body))


def decode_speed(raw):
    """servo present-speed register (int16 as received): bit15 = direction, 15 bit magnitude"""
    u = raw & 0xFFFF
    mag = u & 0x7FFF
    return -mag if u & 0x8000 else mag


def decode_load(raw):
    """servo present-load register: bit10 = direction, 0..1000 = 0.1 % of the stall torque"""
    u = raw & 0xFFFF
    mag = u & 0x3FF
    return -mag if u & 0x400 else mag


def unpack_vxy(v):
    vx = (v >> 8)
    vyaw = v & 0xFF
    if vyaw >= 128:
        vyaw -= 256
    return vx, vyaw


# ---------------------------------------------------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------------------------------------------------
class Recording:
    def __init__(self):
        self.meta = {}
        self.kind = "?"
        self.begin = {}
        self.end = {}
        self.bad_lines = 0
        self.n_events = 0
        self.marks = []                         # (t_s, text)
        self.zero = [512, 512, 512]
        self.mask = [1, 1, -1]
        self.g = 9.8
        self.acc_lsb_per_g = 16384.0
        self.gyro_lsb_per_dps = 32.0
        self.up_scale = 10000.0
        self._I, self._S, self._C, self._V = [], {1: [], 2: [], 3: []}, [], []
        self.complete = False
        # decoded arrays (filled by finalize)
        self.imu_t = self.acc = self.gyro = self.up = self.imu_flags = self.imu_state = self.imu_vx = self.imu_vyaw = None
        self.servo = {}
        self.cmd_t = self.cmd_p = self.cmd_off = self.cmd_speed = self.cmd_src = None
        self.volt_t = self.volt = None

    # -- parsing -----------------------------------------------------------------------------------------------------
    def feed(self, line):
        line = line.rstrip("\r\n")
        if not line.startswith("@"):
            return
        if line.startswith("@begin dump"):
            self.begin = _kv(line[len("@begin dump"):])
            self.kind = self.begin.get("kind", "?")
            return
        if line.startswith("@meta"):
            rest = line[len("@meta"):]
            if rest.strip().startswith("args="):         # free text: everything after 'args='
                self.meta["args"] = rest.strip()[len("args="):]
            else:
                self.meta.update(_kv(rest))
            return
        if line.startswith("@end dump"):
            self.end = _kv(line[len("@end dump"):])
            self.complete = True
            return
        if line.startswith("@fmt") or line.startswith("@ok") or line.startswith("@err"):
            return
        if len(line) < 3 or line[2] != ",":
            return
        if not check_line(line):
            self.bad_lines += 1
            return
        typ = line[1]
        body = line[3:line.rfind("*")]
        try:
            if typ == "M":
                t, text = body.split(",", 1)
                self.marks.append((int(t) * 1e-6, text.strip()))
            else:
                f = [int(x) for x in body.split(",")]
                if typ == "I" and len(f) == 13:
                    self._I.append(f)
                elif typ == "S" and len(f) == 6:
                    if 1 <= f[1] <= 3:
                        self._S[f[1]].append(f)
                elif typ == "C" and len(f) == 6:
                    self._C.append(f)
                elif typ == "V" and len(f) == 2:
                    self._V.append(f)
                else:
                    self.bad_lines += 1
                    return
            self.n_events += 1
        except ValueError:
            self.bad_lines += 1

    def finalize(self):
        need_numpy()
        m = self.meta
        if "zero" in m:
            self.zero = [int(x) for x in m["zero"].split(",")]
        if "mask" in m:
            self.mask = [int(x) for x in m["mask"].split(",")]
        if "g_mm_s2" in m:
            self.g = int(m["g_mm_s2"]) / 1000.0
        self.acc_lsb_per_g = float(m.get("acc_lsb_per_g", self.acc_lsb_per_g))
        self.gyro_lsb_per_dps = float(m.get("gyro_lsb_per_dps", self.gyro_lsb_per_dps))
        self.up_scale = float(m.get("up_scale", self.up_scale))

        a = np.array(self._I, dtype=np.int64).reshape(-1, 13)
        self.imu_t = a[:, 0] * 1e-6
        self.acc = a[:, 1:4] * (self.g / self.acc_lsb_per_g)
        self.gyro = a[:, 4:7] / self.gyro_lsb_per_dps
        self.up = a[:, 7:10] / self.up_scale
        self.imu_flags = a[:, 10].astype(int)
        self.imu_state = a[:, 11].astype(int)
        vxy = [unpack_vxy(int(v)) for v in a[:, 12]]
        self.imu_vx = np.array([v[0] for v in vxy], dtype=float)
        self.imu_vyaw = np.array([v[1] for v in vxy], dtype=float)

        for i in (1, 2, 3):
            s = np.array(self._S[i], dtype=np.int64).reshape(-1, 6)
            pos = s[:, 2].astype(float)
            self.servo[i] = {
                "t": s[:, 0] * 1e-6,
                "pos": pos,
                "off": self.mask[i - 1] * (pos - self.zero[i - 1]) + 0.0,       # "+ 0.0" turns -0.0 into 0.0
                "spd": np.array([decode_speed(int(v)) for v in s[:, 3]], dtype=float),
                "load": np.array([decode_load(int(v)) for v in s[:, 4]], dtype=float),
                "lat_ms": s[:, 5] * 1e-3,
            }
        c = np.array(self._C, dtype=np.int64).reshape(-1, 6)
        self.cmd_t = c[:, 0] * 1e-6
        self.cmd_src = c[:, 1].astype(int)
        self.cmd_p = c[:, 2:5].astype(float)
        self.cmd_speed = c[:, 5].astype(float)
        self.cmd_off = (np.stack([self.mask[i] * (self.cmd_p[:, i] - self.zero[i]) for i in range(3)], axis=1) + 0.0) if len(c) else np.zeros((0, 3))
        v = np.array(self._V, dtype=np.int64).reshape(-1, 2)
        self.volt_t = v[:, 0] * 1e-6
        self.volt = v[:, 1] * 0.01
        return self

    @property
    def duration(self):
        ts = [x[-1] for x in (self.imu_t, self.cmd_t, self.volt_t) if x is not None and len(x)]
        ts += [self.servo[i]["t"][-1] for i in (1, 2, 3) if len(self.servo[i]["t"])]
        lo = [x[0] for x in (self.imu_t,) if x is not None and len(x)]
        return (max(ts) - (lo[0] if lo else 0.0)) if ts else 0.0


def _kv(text):
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def load(path_or_lines):
    """Parse a .tlog file (path) or an iterable of lines into a finalized Recording."""
    rec = Recording()
    if isinstance(path_or_lines, str):
        with open(path_or_lines, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                rec.feed(line)
    else:
        for line in path_or_lines:
            rec.feed(line)
    return rec.finalize()


# ---------------------------------------------------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------------------------------------------------
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return ("%." + str(nd) + "f") % x


def _rate(t):
    if len(t) < 3:
        return 0.0
    dt = np.diff(t)
    med = float(np.median(dt))
    return 1.0 / med if med > 0 else 0.0


def _window(t, lo, hi):
    return (t >= lo) & (t < hi)


def goertzel_scan(t, x, f_lo=0.3, f_hi=8.0, n=200):
    """Dominant frequency of a (possibly unevenly sampled) signal by a least-squares sine scan. Returns (f, amplitude)."""
    if len(t) < 20:
        return None, None
    x = x - np.mean(x)
    best = (None, 0.0)
    for f in np.linspace(f_lo, f_hi, n):
        w = 2 * math.pi * f
        c, s = np.cos(w * t), np.sin(w * t)
        a = 2.0 * np.mean(x * s)
        b = 2.0 * np.mean(x * c)
        amp = math.hypot(a, b)
        if amp > best[1]:
            best = (float(f), float(amp))
    return best


def tilt_from_mean(acc, t, smooth_s=0.1):
    """angle [deg] between the (smoothed) gravity direction and its mean direction: how much the body rocks."""
    if len(acc) < 5:
        return np.zeros(len(acc))
    rate = _rate(t) or 100.0
    n = max(1, int(round(smooth_s * rate)))
    kern = np.ones(n) / n
    sm = np.stack([np.convolve(acc[:, k], kern, mode="same") for k in range(3)], axis=1)
    mean = sm.mean(axis=0)
    mean /= np.linalg.norm(mean) + 1e-9
    nrm = np.linalg.norm(sm, axis=1) + 1e-9
    cosang = np.clip((sm @ mean) / nrm, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


AXES = ["x", "y", "z"]


def _axis_name(v):
    k = int(np.argmax(np.abs(v)))
    return ("+" if v[k] >= 0 else "-") + AXES[k]


# ---------------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------------
def summarize(rec):
    out = []
    out.append("kind=%s  duration=%.2f s  events=%d  bad_lines=%d  device_dropped=%s  complete=%s" % (
        rec.kind, rec.duration, rec.n_events, rec.bad_lines, rec.begin.get("dropped", "?"), rec.complete))
    if rec.meta.get("args", "-") != "-":
        out.append("test arguments: %s" % rec.meta["args"])
    if rec.end:
        exp = rec.end.get("events")
        if exp is not None and int(exp) != rec.n_events:
            out.append("note: device says it sent %s events, %d parsed fine (%d damaged lines)" % (exp, rec.n_events, rec.bad_lines))
    if len(rec.imu_t):
        gaps = np.diff(rec.imu_t)
        out.append("IMU    : %5d samples, %.1f Hz, largest gap %.0f ms" % (len(rec.imu_t), _rate(rec.imu_t), (gaps.max() if len(gaps) else 0) * 1e3))
    for i in (1, 2, 3):
        s = rec.servo[i]
        if len(s["t"]) == 0:
            out.append("servo %d: NO read-back received" % i)
            continue
        lat = s["lat_ms"][s["lat_ms"] > 0]
        out.append("servo %d: %5d replies, %.1f Hz, reply latency %s ms, pos range %d..%d (offset %+.0f..%+.0f), |load| mean %.0f max %.0f" % (
            i, len(s["t"]), _rate(s["t"]), _fmt(float(np.median(lat)) if len(lat) else None, 2), s["pos"].min(), s["pos"].max(),
            s["off"].min(), s["off"].max(), np.mean(np.abs(s["load"])), np.max(np.abs(s["load"]))))
    if len(rec.volt):
        out.append("servo bus voltage: min %.2f V  mean %.2f V  max %.2f V (%d readings)" % (rec.volt.min(), rec.volt.mean(), rec.volt.max(), len(rec.volt)))
    if len(rec.cmd_t):
        out.append("commands: %d logged, servo-1 offset range %+.0f..%+.0f" % (len(rec.cmd_t), rec.cmd_off[:, 0].min(), rec.cmd_off[:, 0].max()))
    for t, txt in rec.marks:
        out.append("marker @ %.3f s: %s" % (t, txt))
    return out


def analyze_still(rec, t0=None, t1=None):
    """Sensor statistics of a still robot (test hold)."""
    out = []
    if not len(rec.imu_t):
        return ["no IMU data"]
    lo = rec.imu_t[0] if t0 is None else t0
    hi = rec.imu_t[-1] + 1 if t1 is None else t1
    m = _window(rec.imu_t, lo, hi)
    a, g = rec.acc[m], rec.gyro[m]
    amean, astd = a.mean(axis=0), a.std(axis=0)
    anorm = np.linalg.norm(a, axis=1)
    raw_pre = a * (rec.acc_lsb_per_g / rec.g)
    stuck_pre = [k for k in range(3) if abs(raw_pre[:, k].mean()) >= 32700 and raw_pre[:, k].std() < 1.0]
    amean_ok = np.array([0.0 if k in stuck_pre else amean[k] for k in range(3)])
    gdir = amean_ok / (np.linalg.norm(amean_ok) + 1e-9)
    out.append("accelerometer mean (m/s^2): x %+.3f  y %+.3f  z %+.3f   |a| = %.3f (expect 9.8)" % (amean[0], amean[1], amean[2], np.linalg.norm(amean)))
    out.append("accelerometer noise std : x %.3f  y %.3f  z %.3f m/s^2   (|a| std %.3f)" % (astd[0], astd[1], astd[2], anorm.std()))
    out.append("gravity points along sensor axis %s   (direction cosines x %+.3f y %+.3f z %+.3f%s)" % (
        _axis_name(amean_ok), gdir[0], gdir[1], gdir[2], ", stuck axis ignored" if stuck_pre else ""))
    gm, gs = g.mean(axis=0), g.std(axis=0)
    out.append("gyro bias (deg/s)       : x %+.3f  y %+.3f  z %+.3f   noise std x %.3f y %.3f z %.3f" % (gm[0], gm[1], gm[2], gs[0], gs[1], gs[2]))
    raw_still = rec.acc[m] * (rec.acc_lsb_per_g / rec.g)
    stuck_axes = [AXES[k] for k in range(3) if abs(raw_still[:, k].mean()) >= 32700 and raw_still[:, k].std() < 1.0]
    if stuck_axes:
        out.append("WARNING: accelerometer axis %s is stuck at the full-scale code: that channel is dead (|a| above is meaningless)" % ",".join(stuck_axes))
    elif abs(np.linalg.norm(amean) - rec.g) > 0.6:
        out.append("WARNING: |a| differs from 9.8 m/s^2 by more than 0.6 -> robot not still, or the accelerometer scale is off")
    if np.abs(gm).max() > 3.0:
        kk = int(np.argmax(np.abs(gm)))
        if gs[kk] < 0.8:
            out.append("WARNING: gyro %s axis has a steady offset of %+.1f deg/s with little noise: a real zero offset of that gyro channel, not motion" % (AXES[kk], gm[kk]))
        else:
            out.append("WARNING: gyro bias above 3 deg/s -> robot was moving during the capture")
    flagged = (rec.imu_flags[m] & 1).mean() if m.any() else 0
    if flagged > 0:
        up = rec.up[m]
        upm = up.mean(axis=0)
        out.append("attitude filter 'up' (robot frame): x %+.3f  y %+.3f  z %+.3f  (calibrated=%s)" % (upm[0], upm[1], upm[2], bool(rec.imu_flags[m][-1] & 2)))
    if len(rec.volt):
        out.append("servo bus voltage while still: %.2f V" % float(np.mean(rec.volt)))
    return out


def _segments_from_cmd(rec, sid, min_step):
    """command change points of servo `sid` (offset units) -> list of (t_cmd, level_before, level_after, t_end)"""
    if not len(rec.cmd_t):
        return []
    off = rec.cmd_off[:, sid - 1]
    seg = []
    prev = off[0]
    changes = []
    for k in range(1, len(off)):
        if abs(off[k] - prev) >= min_step:
            changes.append((rec.cmd_t[k], prev, off[k]))
            prev = off[k]
        elif abs(off[k] - prev) > 0:
            prev = off[k] if abs(off[k] - prev) < min_step * 0.3 else prev
    for n, (t, a, b) in enumerate(changes):
        t_end = changes[n + 1][0] if n + 1 < len(changes) else (rec.cmd_t[-1] + 0.4)
        seg.append((t, a, b, t_end))
    return seg


def step_metrics(rec, sid, amp):
    """per commanded step of servo `sid`: list of dicts (delay / rise time / overshoot / settling / peak speed / final error / peak load)"""
    s = rec.servo[sid]
    rows = []
    segs = _segments_from_cmd(rec, sid, 0.4 * amp)
    for n, (tc, a, b, t_end) in enumerate(segs):
        row = {"n": n + 1, "from": a, "to": b, "t_cmd": tc, "note": ""}
        rows.append(row)
        t, x = s["t"], s["off"]
        pre = _window(t, tc - 0.15, tc)
        x0 = float(np.median(x[pre])) if pre.sum() >= 1 else float(x[max(0, np.searchsorted(t, tc) - 1)])
        dwell = _window(t, tc, t_end)
        if dwell.sum() < 4:
            row["note"] = "too few samples"
            continue
        tail = _window(t, tc + 0.7 * (t_end - tc), t_end)
        xf = float(np.median(x[tail])) if tail.sum() else float(x[dwell][-1])
        d = xf - x0
        if abs(d) < 0.15 * abs(b - a):
            row["note"] = "servo barely moved (%+.1f -> %+.1f): blocked leg or wrong servo?" % (x0, xf)
            continue
        ts, xs = t[dwell], x[dwell]
        ys = (xs - x0) / d

        def cross(level):
            idx = np.where(ys >= level)[0]
            if len(idx) == 0:
                return None
            k = idx[0]
            if k == 0:
                return ts[0]
            return ts[k - 1] + (level - ys[k - 1]) * (ts[k] - ts[k - 1]) / max(ys[k] - ys[k - 1], 1e-9)

        t10, t90 = cross(0.1), cross(0.9)
        outside = np.where(np.abs(ys - 1.0) > 0.05)[0]
        if len(outside) == 0:
            settle = 0.0
        elif outside[-1] + 1 < len(ts):
            settle = (ts[outside[-1] + 1] - tc) * 1e3
        else:
            settle = None                                   # never settled inside the window
        vmax = 0.0
        for k in range(len(ts)):                            # peak speed from >= 8 ms spans
            j = int(np.searchsorted(ts, ts[k] + 0.008))
            if j < len(ts):
                vmax = max(vmax, abs((xs[j] - xs[k]) / (ts[j] - ts[k])))
        row.update({
            "delay_ms": (t10 - tc) * 1e3 if t10 is not None else None,
            "rise_ms": (t90 - t10) * 1e3 if (t90 is not None and t10 is not None) else None,
            "overshoot_pct": max(0.0, float(ys.max()) - 1.0) * 100.0,
            "settle_ms": settle,
            "vmax": vmax,
            "final_err": xf - b,
            "load_peak": float(np.abs(s["load"][dwell]).max()) if dwell.any() else 0.0,
        })
    return rows


def analyze_step(rec, sid=None, amp=None):
    out = []
    args = _kv(rec.meta.get("args", "").replace(",", " ")) if rec.meta.get("args", "-") != "-" else {}
    if sid is None:
        sid = int(args.get("id", 1))
    if amp is None:
        amp = float(args.get("amp", 0)) or None
    s = rec.servo[sid]
    if len(s["t"]) < 5:
        return ["servo %d returned no read-back samples - check cable / servo ID" % sid]
    if amp is None:
        amp = float(np.max(np.abs(rec.cmd_off[:, sid - 1])))
    rows = step_metrics(rec, sid, amp)
    out.append("servo %d step response (commands are WritePos units; 1 count = %.3f deg of servo travel)" % (sid, DEG_PER_COUNT))
    out.append("  #  cmd step     delay   rise10-90  overshoot  settle(5%)  peak speed            final err  peak|load|")
    for r in rows:
        head = "  %d  %+5.0f->%+5.0f" % (r["n"], r["from"], r["to"])
        if r["note"]:
            out.append("%s  %s" % (head, r["note"]))
            continue
        out.append("%s  %6s ms  %6s ms  %7.1f %%  %7s ms  %5.0f cnt/s (%3.0f deg/s)  %+6.1f cnt  %6.0f" % (
            head, _fmt(r["delay_ms"], 1), _fmt(r["rise_ms"], 1), r["overshoot_pct"], _fmt(r["settle_ms"], 0), r["vmax"], r["vmax"] * DEG_PER_COUNT,
            r["final_err"], r["load_peak"]))
    if not rows:
        out.append("  no command steps found in the log (C events missing?)")
    dt = np.diff(s["t"])
    if len(dt):
        lat = s["lat_ms"][s["lat_ms"] > 0]
        out.append("  read-back sampling: median %.1f ms between samples (%.0f Hz), reply latency median %s ms" % (
            np.median(dt) * 1e3, 1.0 / np.median(dt), _fmt(float(np.median(lat)) if len(lat) else None, 2)))
    if len(rec.imu_t):
        out.append("  IMU during the test: peak |gyro| %.0f deg/s, peak |acc - mean| %.2f m/s^2" % (
            np.abs(rec.gyro).max(), np.abs(rec.acc - rec.acc.mean(axis=0)).max()))
    return out


def analyze_sine(rec, sid=None, freq=None):
    out = []
    args = _kv(rec.meta.get("args", "").replace(",", " ")) if rec.meta.get("args", "-") != "-" else {}
    if sid is None:
        sid = int(args.get("id", 1))
    if freq is None:
        freq = float(args.get("freq_mhz", 1000)) / 1000.0
    s = rec.servo[sid]
    if len(s["t"]) < 10 or len(rec.cmd_t) < 10:
        return ["not enough samples for the frequency response"]
    cmd = rec.cmd_off[:, sid - 1]
    active = np.where(np.abs(cmd) > 0.02 * np.abs(cmd).max())[0]
    t_lo, t_hi = rec.cmd_t[active[0]], rec.cmd_t[active[-1]]
    t_lo += 1.0 / freq          # skip the first period (start-up transient)
    if t_hi - t_lo < 1.0 / freq:
        t_lo = rec.cmd_t[active[0]]
    w = 2 * math.pi * freq

    def fit(t, x):
        m = (t >= t_lo) & (t <= t_hi)
        t, x = t[m], x[m]
        A = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return math.hypot(coef[0], coef[1]), math.atan2(coef[1], coef[0]), coef[2], len(t)

    ac, pc, _, nc = fit(rec.cmd_t, cmd)
    ax, px, off, nx = fit(s["t"], s["off"])
    gain = ax / ac if ac > 1e-6 else float("nan")
    lag = math.degrees(pc - px)
    lag = (lag + 180.0) % 360.0 - 180.0
    out.append("servo %d at %.2f Hz: command amplitude %.1f counts (%d samples), response amplitude %.1f counts (%d samples), offset %+.1f" % (sid, freq, ac, nc, ax, nx, off))
    out.append("  gain = %.3f (%.1f dB)   phase lag = %.1f deg  (= %.1f ms of delay)" % (gain, 20 * math.log10(gain) if gain > 0 else float("nan"), lag, lag / 360.0 / freq * 1e3))
    return out


def analyze_gait(rec):
    out = []
    if len(rec.imu_t) < 30:
        return ["not enough IMU samples for a gait analysis"]
    g = rec.gyro
    gm = np.linalg.norm(g, axis=1)
    out.append("gyro RMS (deg/s): x %.1f  y %.1f  z %.1f   peak |gyro| %.0f deg/s" % (*np.sqrt((g ** 2).mean(axis=0)), gm.max()))
    best = (None, 0.0, None)
    for k in range(3):                                   # per axis: |gyro| would show the doubled frequency
        f, amp = goertzel_scan(rec.imu_t, g[:, k])
        if f and amp > best[1]:
            best = (f, amp, AXES[k])
    if best[0]:
        out.append("dominant body-motion frequency: %.2f Hz on gyro %s (amplitude %.1f deg/s); the classic gait runs at 1.25 Hz" % (best[0], best[2], best[1]))
    tilt = tilt_from_mean(rec.acc, rec.imu_t)
    out.append("body rocking (angle between smoothed gravity and its mean direction): mean %.1f deg, 95%% %.1f deg, max %.1f deg" % (
        tilt.mean(), np.percentile(tilt, 95), tilt.max()))
    an = np.linalg.norm(rec.acc, axis=1)
    out.append("|acc|: mean %.2f  min %.2f  max %.2f m/s^2  (spikes above ~20 m/s^2 are impacts / falls)" % (an.mean(), an.min(), an.max()))
    if tilt.max() > 60:
        out.append("WARNING: the body tilted more than 60 deg from its average attitude - probably fell over")
    for i in (1, 2, 3):
        sv = rec.servo[i]
        if len(sv["t"]):
            out.append("servo %d: |load| mean %.0f  p95 %.0f  max %.0f   position offset range %+.0f..%+.0f" % (
                i, np.abs(sv["load"]).mean(), np.percentile(np.abs(sv["load"]), 95), np.abs(sv["load"]).max(), sv["off"].min(), sv["off"].max()))
    if len(rec.volt):
        out.append("servo bus voltage: min %.2f V, mean %.2f V (lowest value shows the sag under load)" % (rec.volt.min(), rec.volt.mean()))
    return out


STATE_NAMES = ["STAND", "WALK_F", "WALK_B", "TURN_L", "TURN_R", "FALLEN", "GETUP1", "GETUP2A", "GETUP2B", "GETUP2C", "FAILED", "GETUP_NN", "MOTION"]   # tars_state_t in tars_ctrl.h


def analyze_learned(rec):
    """timeline of the learned supervisor (states logged in the IMU events while the learned mode is on)"""
    out = []
    st = rec.imu_state
    on = st >= 0
    if not len(st) or not on.any():
        return out
    t = rec.imu_t
    name = lambda k: STATE_NAMES[k] if 0 <= k < len(STATE_NAMES) else "?%d" % k
    segs = []                                   # (t_start, t_end, state)
    start = 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[start]:
            if st[start] >= 0:
                segs.append((t[start], t[min(i, len(st) - 1)], int(st[start])))
            start = i
    out.append("learned mode was on for %.1f s (dry run: %s)" % (float(on.sum()) * (t[-1] - t[0]) / max(len(t) - 1, 1), "yes" if (rec.imu_flags[on] & 8).all() else "no"))
    out.append("state timeline: " + "; ".join("%.1f-%.1fs %s" % (a, b, name(k)) for a, b, k in segs[:30]) + (" ..." if len(segs) > 30 else ""))
    entries = lambda k: sum(1 for i, (_, _, kk) in enumerate(segs) if kk == k and (i == 0 or segs[i - 1][2] != k))
    falls, getups, fails = entries(5), entries(6) + entries(11), entries(10)         # attempts = keyframe (GETUP1) + learned network (GETUP_NN)
    out.append("falls detected: %d   get-up attempts: %d   FAILED: %d" % (falls, getups, fails))
    if entries(12):
        out.append("show motions performed: %d (state MOTION)" % entries(12))
    for i, (a, b, k) in enumerate(segs):
        if k == 5 and i + 1 < len(segs):
            nxt = [seg for seg in segs[i + 1:] if seg[2] in (0, 10)]
            if nxt:
                out.append("  fall at %.1f s -> %s again at %.1f s (%.1f s later)" % (a, name(nxt[0][2]), nxt[0][0], nxt[0][0] - a))
    tot = max(sum(b - a for a, b, _ in segs), 1e-9)
    frac = {}
    for a, b, k in segs:
        frac[k] = frac.get(k, 0.0) + (b - a)
    out.append("time in state: " + ", ".join("%s %.0f%%" % (name(k), 100 * v / tot) for k, v in sorted(frac.items(), key=lambda kv: -kv[1])))
    m = on & ((rec.imu_flags & 1) > 0)
    if m.any():
        up = rec.up[m]
        pitch = np.degrees(np.arcsin(np.clip(-up[:, 0], -1, 1)))                # + = nose down
        out.append("attitude (filter): body pitch min %+.1f / max %+.1f deg (+ = leaning forward), lowest up_z %.2f" % (pitch.min(), pitch.max(), up[:, 2].min()))
    return out


def analyze(rec):
    """Text report for a recording of any kind."""
    lines = ["== summary =="] + summarize(rec)
    if rec.kind == "hold":
        lines += ["", "== robot still =="] + analyze_still(rec)
    elif rec.kind == "step":
        lines += ["", "== step response =="] + analyze_step(rec)
        lines += ["", "== robot before the test (first 0.3 s) =="] + analyze_still(rec, rec.imu_t[0] if len(rec.imu_t) else 0, (rec.imu_t[0] if len(rec.imu_t) else 0) + 0.3)
    elif rec.kind == "sine":
        lines += ["", "== frequency response =="] + analyze_sine(rec)
    else:
        lines += ["", "== motion =="] + analyze_gait(rec)
    learned = analyze_learned(rec)
    if learned:
        lines += ["", "== learned behaviour =="] + learned
    return lines


def health_check(rec):
    """checklist for a `test hold` recording (robot still): list of (status, text), status in 'ok' / 'warn' / 'fail'"""
    items = []

    def add(status, text):
        items.append((status, text))

    if len(rec.imu_t) < 20:
        add("fail", "no IMU samples received")
    else:
        rate = _rate(rec.imu_t)
        add("ok" if 85 <= rate <= 110 else "warn", "IMU stream %.0f Hz (expected ~100)" % rate)
        if (rec.imu_flags & 16).any():
            add("fail", "IMU not initialised (I2C problem?) - the accelerometer values are meaningless")
        am = rec.acc.mean(axis=0)
        norm = float(np.linalg.norm(am))
        raw = rec.acc * (rec.acc_lsb_per_g / rec.g)
        stuck = [k for k in range(3) if abs(raw[:, k].mean()) >= 32700 and raw[:, k].std() < 1.0]
        for k in stuck:
            add("fail", "accelerometer %s axis is stuck at the full-scale code (%d LSB, no noise): that sensor channel is dead - 'imu regs' / "
                        "'imu fs 1' show whether it follows the range; the other two axes can still be used" % (AXES[k], int(round(raw[:, k].mean()))))
        if (rec.imu_flags & 32).any() and not stuck:
            add("warn", "an accelerometer axis reached the 16-bit limit during the recording (impact, or a stuck channel)")
        ok_mag = abs(norm - rec.g) < 0.5
        if not ok_mag and stuck:
            add("warn", "|acceleration| = %.2f m/s^2 is meaningless while an axis is stuck (x/y/z without the stuck axis: %.2f)" % (
                norm, float(np.linalg.norm([am[k] for k in range(3) if k not in stuck]))))
        else:
            add("ok" if ok_mag else "fail", "|acceleration| = %.2f m/s^2 (expected 9.8 when the robot stands still)" % norm)
        am_ok = np.array([0.0 if k in stuck else am[k] for k in range(3)])
        add("ok", "gravity points along sensor axis %s (direction cosines %+.2f %+.2f %+.2f%s)" % (
            (_axis_name(am_ok),) + tuple(am_ok / (np.linalg.norm(am_ok) + 1e-9)) + (", stuck axis ignored" if stuck else "",)))
        gb = rec.gyro.mean(axis=0)
        gs = rec.gyro.std(axis=0)
        if np.abs(gb).max() < 3.0:
            add("ok", "gyro bias %+.2f %+.2f %+.2f deg/s" % tuple(gb))
        else:
            k = int(np.argmax(np.abs(gb)))
            if gs[k] < 0.8:
                add("warn", "gyro %s axis has a steady zero offset of %+.1f deg/s (almost no noise, so it is not motion): the attitude filter / yaw integral "
                            "drift unless this offset is subtracted" % (AXES[k], gb[k]))
            else:
                add("warn", "gyro bias %+.2f %+.2f %+.2f deg/s (|bias| should be below 3: was the robot really still?)" % tuple(gb))
        noise = rec.acc.std(axis=0)
        add("ok" if noise.max() < 0.3 else "warn", "accelerometer noise %.3f m/s^2 rms (high values mean vibration: servos fighting a load?)" % float(noise.max()))
    for i in (1, 2, 3):
        sv = rec.servo[i]
        if len(sv["t"]) < 10:
            add("fail", "servo %d: no replies (cable / ID / power?)" % i)
            continue
        r = _rate(sv["t"])
        lat = sv["lat_ms"][sv["lat_ms"] > 0]
        pos_ok = 100 < sv["pos"].mean() < 900
        add("ok" if (r > 30 and pos_ok) else "warn", "servo %d replies at %.0f Hz, latency %.2f ms, position %d (offset %+.0f from the stored zero)" % (
            i, r, float(np.median(lat)) if len(lat) else float("nan"), sv["pos"].mean(), sv["off"].mean()))
        if abs(sv["off"].mean()) > 25:
            add("warn", "servo %d is %+.0f counts away from the stored zero: not in the calibrated rest pose, or zero calibration is stale" % (i, sv["off"].mean()))
    if len(rec.volt):
        v = float(rec.volt.mean())
        add("ok" if 6.6 <= v <= 8.5 else "warn", "servo bus voltage %.2f V (2S Li-ion: 6.6 empty .. 8.4 full)" % v)
    else:
        add("warn", "no servo voltage reading")
    return items


# ---------------------------------------------------------------------------------------------------------------------
# flat CSV (zero-order hold onto the IMU time base)
# ---------------------------------------------------------------------------------------------------------------------
def to_csv(rec, f):
    need_numpy()
    t = rec.imu_t
    cols = ["t_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "up_x", "up_y", "up_z", "state", "vx", "vyaw"]
    cols += ["pos1", "pos2", "pos3", "off1", "off2", "off3", "load1", "load2", "load3", "spd1", "spd2", "spd3"]
    cols += ["cmd_off1", "cmd_off2", "cmd_off3", "volt"]
    f.write(",".join(cols) + "\n")

    def zoh(tt, vals):
        if len(tt) == 0:
            return np.full(len(t), np.nan)
        idx = np.searchsorted(tt, t, side="right") - 1
        out = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan)
        return out

    sv = {i: rec.servo[i] for i in (1, 2, 3)}
    series = []
    series += [rec.acc[:, k] for k in range(3)] + [rec.gyro[:, k] for k in range(3)] + [rec.up[:, k] for k in range(3)]
    series += [rec.imu_state.astype(float), rec.imu_vx, rec.imu_vyaw]
    series += [zoh(sv[i]["t"], sv[i]["pos"]) for i in (1, 2, 3)]
    series += [zoh(sv[i]["t"], sv[i]["off"]) for i in (1, 2, 3)]
    series += [zoh(sv[i]["t"], sv[i]["load"]) for i in (1, 2, 3)]
    series += [zoh(sv[i]["t"], sv[i]["spd"]) for i in (1, 2, 3)]
    series += [zoh(rec.cmd_t, rec.cmd_off[:, k]) if len(rec.cmd_t) else np.full(len(t), np.nan) for k in range(3)]
    series += [zoh(rec.volt_t, rec.volt)]
    for n in range(len(t)):
        row = ["%.4f" % t[n]] + ["" if np.isnan(s[n]) else ("%.4f" % (s[n] + 0.0)).rstrip("0").rstrip(".") for s in series]
        f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# HTML report with dependency-free SVG charts
# ---------------------------------------------------------------------------------------------------------------------
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _decimate(t, y, n=1400):
    if len(t) <= n:
        return t, y
    k = int(math.ceil(len(t) / n))
    return t[::k], y[::k]


def svg_chart(title, series, ylabel="", width=920, height=230, step=False, marks=()):
    """series: [(name, t, y)] -> inline SVG string."""
    ml, mr, mt, mb = 58, 130, 26, 30
    pw, ph = width - ml - mr, height - mt - mb
    xs = [x for _, t, y in series if len(t) for x in (t[0], t[-1])]
    ys = [v for _, t, y in series if len(y) for v in (float(np.nanmin(y)), float(np.nanmax(y)))]
    if not xs or not ys:
        return "<p>%s: no data</p>" % html.escape(title)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    pad = 0.06 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad
    if x1 - x0 < 1e-9:
        x1 = x0 + 1

    def sx(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return mt + (1 - (v - y0) / (y1 - y0)) * ph

    parts = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s" style="max-width:%dpx">' % (width, height, html.escape(title), width)]
    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="currentColor" stroke-opacity=".25"/>' % (ml, mt, pw, ph))
    for k in range(6):
        v = y0 + (y1 - y0) * k / 5
        parts.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="currentColor" stroke-opacity=".12"/>' % (ml, ml + pw, sy(v), sy(v)))
        parts.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end" fill="currentColor">%s</text>' % (ml - 4, sy(v) + 3, ("%.3g" % v)))
    for k in range(9):
        v = x0 + (x1 - x0) * k / 8
        parts.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="currentColor">%.2f</text>' % (sx(v), mt + ph + 14, v))
    parts.append('<text x="%d" y="14" font-size="12" font-weight="600" fill="currentColor">%s</text>' % (ml, html.escape(title)))
    parts.append('<text x="%d" y="%d" font-size="10" text-anchor="end" fill="currentColor">t [s]</text>' % (ml + pw, height - 4))
    if ylabel:
        parts.append('<text x="4" y="%d" font-size="10" fill="currentColor">%s</text>' % (mt + 8, html.escape(ylabel)))
    for tm, txt in marks:
        if x0 <= tm <= x1:
            parts.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#888" stroke-dasharray="3 3"/>' % (sx(tm), sx(tm), mt, mt + ph))
    for n, (name, t, y) in enumerate(series):
        if not len(t):
            continue
        t, y = _decimate(np.asarray(t), np.asarray(y))
        col = PALETTE[n % len(PALETTE)]
        pts = []
        for a, b in zip(t, y):
            if np.isnan(b):
                continue
            if step and pts:
                pts.append("%.1f,%.1f" % (sx(a), float(pts[-1].split(",")[1])))
            pts.append("%.1f,%.1f" % (sx(a), sy(b)))
        parts.append('<polyline fill="none" stroke="%s" stroke-width="1.4" points="%s"/>' % (col, " ".join(pts)))
        parts.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="%s" stroke-width="2.5"/>' % (ml + pw + 10, ml + pw + 28, mt + 14 + 16 * n, mt + 14 + 16 * n, col))
        parts.append('<text x="%d" y="%d" font-size="11" fill="currentColor">%s</text>' % (ml + pw + 32, mt + 18 + 16 * n, html.escape(name)))
    parts.append("</svg>")
    return "".join(parts)


def report_html(rec, title="Tars recording"):
    need_numpy()
    charts = []
    marks = rec.marks
    if len(rec.imu_t):
        charts.append(svg_chart("accelerometer [m/s^2]", [("x", rec.imu_t, rec.acc[:, 0]), ("y", rec.imu_t, rec.acc[:, 1]), ("z", rec.imu_t, rec.acc[:, 2])], marks=[(m[0], m[1]) for m in marks]))
        charts.append(svg_chart("gyro [deg/s]", [("x", rec.imu_t, rec.gyro[:, 0]), ("y", rec.imu_t, rec.gyro[:, 1]), ("z", rec.imu_t, rec.gyro[:, 2])]))
        if (rec.imu_flags & 1).any():
            m = (rec.imu_flags & 1) > 0
            charts.append(svg_chart("attitude filter 'up' vector (robot frame)", [("x", rec.imu_t[m], rec.up[m, 0]), ("y", rec.imu_t[m], rec.up[m, 1]), ("z", rec.imu_t[m], rec.up[m, 2])]))
    for i in (1, 2, 3):
        s = rec.servo[i]
        ser = []
        if len(s["t"]):
            ser.append(("feedback", s["t"], s["off"]))
        if len(rec.cmd_t):
            ser.append(("command", rec.cmd_t, rec.cmd_off[:, i - 1]))
        if ser:
            charts.append(svg_chart("servo %d position offset from zero [counts]" % i, ser, step=False))
    loads = [("servo %d" % i, rec.servo[i]["t"], rec.servo[i]["load"]) for i in (1, 2, 3) if len(rec.servo[i]["t"])]
    if loads:
        charts.append(svg_chart("servo load [0.1 % of stall torque, signed]", loads))
    if len(rec.volt):
        charts.append(svg_chart("servo bus voltage [V]", [("V", rec.volt_t, rec.volt)], step=True))
    body = "\n".join("<div class='c'>%s</div>" % c for c in charts)
    text = html.escape("\n".join(analyze(rec)))
    return """<!doctype html><html><head><meta charset="utf-8"><title>%s</title>
<style>:root{color-scheme:light dark} body{font:14px/1.45 -apple-system,Segoe UI,sans-serif;margin:24px auto;max-width:980px;padding:0 12px}
pre{background:rgba(127,127,127,.12);padding:12px;border-radius:6px;overflow:auto;font-size:12px} .c{margin:10px 0}</style></head>
<body><h1>%s</h1><pre>%s</pre>%s</body></html>""" % (html.escape(title), html.escape(title), text, body)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: tlog.py FILE.tlog")
    print("\n".join(analyze(load(sys.argv[1]))))
