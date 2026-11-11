#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_flow.py - guided bring-up of the learned behaviours on a real Tars (used by `tars_serial.py setup`).

  0. pre-flight : 3 servos answer, battery voltage, IMU channels alive
  1. IMU axes   : cal 0 (upright) / cal 1 (nose down ~30 deg), then check the attitude estimate in both postures
  2. servo directions : move each servo a little (robot in the hand, legs free), ask what happened, store the mapping
  3. dry run    : learn dry - the state machine runs, the servos do not move

Nothing in here moves the robot except the three small `pose` tests of stage 2 (+-40 counts = 12 degrees, 4 s each, you hold the
robot in the air) and the two `selftest` poses that confirm the stored mapping.  Ctrl-C = emergency stop.
All prompts are plain text so the flow can be driven by a scripted IO object in the tests.
"""
import re
import time

POSE_COUNTS = 40
TIME_SCALE = 1.0          # the tests run the simulator faster than real time and shrink the waits by the same factor
RE_UP = re.compile(r"\bup=(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)")
RE_ACC = re.compile(r"acc=(-?\d+),(-?\d+),(-?\d+)")
RE_ST = re.compile(r"\bst=([A-Z0-9_]+)")
RE_OFF = re.compile(r"\boff=(-?\d+),(-?\d+),(-?\d+)")


class ConsoleIO:
    """interactive console"""

    def say(self, text=""):
        print(text)

    def pause(self, text):
        self.say(text)
        try:
            input("    [回车继续 / press Enter] ")
        except EOFError:
            raise SystemExit("no terminal input: this step needs you at the keyboard")

    def choose(self, text, options):
        self.say(text)
        for i, o in enumerate(options, 1):
            self.say("    [%d] %s" % (i, o))
        while True:
            try:
                raw = input("    选择 / choose 1-%d: " % len(options)).strip()
            except EOFError:
                raise SystemExit("no terminal input: this step needs you at the keyboard")
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw)

    def confirm(self, text):
        try:
            return input(text + " [y/N] ").strip().lower() in ("y", "yes", "是", "好")
        except EOFError:
            return False


class Abort(Exception):
    pass


class Setup:
    def __init__(self, dev, io, verbose=True):
        self.dev = dev
        self.io = io
        self.verbose = verbose
        self.results = {}

    # ---------------------------------------------------------------------------------------------------------------
    def send(self, cmd, timeout=8.0, quiet=False):
        """send one command, return (ok, lines) once '@ok' / '@err' arrived"""
        self.dev.drain()
        self.dev.send(cmd)
        lines = []
        end = time.time() + timeout
        ok = False
        while time.time() < end:
            line = self.dev.get(0.1)
            if not line:
                continue
            if line.startswith("@") and line[2:3] == ",":
                continue
            lines.append(line)
            if line.startswith("@ok"):
                ok = True
                break
            if line.startswith("@err"):
                break
        if self.verbose and not quiet:
            for l in lines:
                if l.startswith("# ") or l.startswith("@err"):
                    self.io.say("      " + l)
        return ok, lines

    def watch(self, seconds, hz=5):
        """run `watch` for a while, return the status lines"""
        self.dev.drain()
        self.dev.send("watch %d" % hz)
        out = []
        end = time.time() + seconds * TIME_SCALE
        while time.time() < end:
            line = self.dev.get(0.1)
            if line.startswith("# w ") or line.startswith("t="):
                out.append(line)
            elif line.startswith("# [rl]"):
                out.append(line)
        self.dev.send("stop")
        t_end = time.time() + 3
        while time.time() < t_end:
            line = self.dev.get(0.1)
            if line.startswith("@ok stop"):
                break
        return out

    @staticmethod
    def up_vectors(lines):
        ups = []
        for l in lines:
            m = RE_UP.search(l)
            if m:
                v = [float(x) for x in m.groups()]
                if abs(v[0]) + abs(v[1]) + abs(v[2]) > 0.01:
                    ups.append(v)
        return ups

    @staticmethod
    def median_vec(vs):
        return [sorted(v[k] for v in vs)[len(vs) // 2] for k in range(3)]

    # ---------------------------------------------------------------------------------------------------------------
    def preflight(self):
        say = self.io.say
        say("== 0/3 检查 / pre-flight ==")
        ok, lines = self.send("scan", 8, quiet=True)
        missing = [l for l in lines if "NO REPLY" in l]
        volt = None
        for l in lines:
            m = re.search(r"voltage: ([0-9.]+) V", l)
            if m:
                volt = float(m.group(1))
        if not ok or missing:
            for l in lines:
                say("      " + l)
            raise Abort("有舵机没有应答（线、ID、供电）/ a servo does not answer")
        say("   3 个舵机都有应答；舵机总线电压 %s V" % ("%.2f" % volt if volt is not None else "?"))
        if volt is not None and volt < 6.9:
            say("   ! 电压偏低（<6.9 V），舵机力量会变小，先充电再测 / low battery")
        ok, lines = self.send("imu raw 4", 10, quiet=True)
        rails = []
        for l in lines:
            m = RE_ACC.search(l)
            if m:
                for k, v in enumerate(m.groups()):
                    if abs(int(v)) >= 32700:
                        rails.append("xyz"[k])
        if not ok:
            raise Abort("读不到 IMU / cannot read the IMU: " + " | ".join(lines[-2:]))
        if rails:
            raise Abort("IMU 加速度计 %s 轴卡死在满量程（通道坏了），学习模式无法使用这台机器 / accelerometer axis %s is stuck" % (",".join(sorted(set(rails))), ",".join(sorted(set(rails)))))
        say("   IMU 三个轴都在正常读数")

    # ---------------------------------------------------------------------------------------------------------------
    def calibrate_imu(self):
        say = self.io.say
        say("")
        say("== 1/3 IMU 轴向标定 / IMU axes ==")
        say("   两步：先竖直站立静止，再向前倾约 30° 静止。每步采样约 1 秒。")
        for attempt in range(3):
            self.io.pause("   第 1 步：把机器人【竖直站立】（脚着地，屏幕朝前），保持不动。")
            ok, lines = self.send("cal 0", 12)
            if ok:
                break
            say("   第 1 步没成功，再来一次（保持静止）。")
        else:
            raise Abort("cal 0 一直失败 / cal 0 keeps failing")
        for attempt in range(3):
            self.io.pause("   第 2 步：把机器人【向屏幕一侧（前）倾斜约 30°】并保持不动（可以靠在东西上，重心别让它倒）。")
            ok, lines = self.send("cal 1", 12)
            if ok and any("calibrated=1" in l for l in lines):
                break
            say("   第 2 步没成功（倾斜角太小或没静止）：再倾斜大一点、更稳一点，再来一次。")
        else:
            raise Abort("cal 1 一直失败 / cal 1 keeps failing")
        say("   标定已保存。验证姿态估计：")
        self.io.pause("   把机器人放回【竖直站立】，保持静止。")
        ups = self.up_vectors(self.watch(3.0))
        if len(ups) < 3:
            raise Abort("姿态滤波器没有启动（机器人没静止？）/ the attitude filter did not start - robot not still?")
        up = self.median_vec(ups[-8:])
        say("   竖直时 up = (%+.2f, %+.2f, %+.2f)   应该约为 (0, 0, +1)" % tuple(up))
        if not (up[2] > 0.95 and abs(up[0]) < 0.2 and abs(up[1]) < 0.2):
            raise Abort("竖直姿态读数不对：标定可能做错了，请重新运行 setup / upright attitude is wrong, run setup again")
        self.io.pause("   再把机器人【向前倾约 30°】并保持静止。")
        ups = self.up_vectors(self.watch(3.0))
        if len(ups) < 3:
            raise Abort("前倾时没读到姿态 / no attitude reading in the tilted pose")
        up = self.median_vec(ups[-8:])
        say("   前倾时 up = (%+.2f, %+.2f, %+.2f)   应该约为 (-0.5, 0, +0.87)：x 为负 = 前倾" % tuple(up))
        if not (-0.8 < up[0] < -0.25):
            raise Abort("前倾方向的符号或大小不对（是不是向后倾了？）/ forward-tilt reading is wrong - did you tilt backwards?")
        self.results["imu"] = True

    # ---------------------------------------------------------------------------------------------------------------
    def _pose(self, c1, c2, c3, secs=4):
        self.send("pose %d %d %d %d" % (c1, c2, c3, secs + 6), 6, quiet=True)
        time.sleep(secs * TIME_SCALE)
        ok, lines = self.send("scan", 8, quiet=True)
        offs = {}
        for l in lines:
            m = re.match(r"# servo (\d): pos=(-?\d+) off=(-?\d+)", l)
            if m:
                offs[int(m.group(1))] = int(m.group(3))
        self.send("release", 4, quiet=True)
        return offs

    def servo_directions(self):
        say = self.io.say
        say("")
        say("== 2/3 舵机方向 / servo directions ==")
        say("   请这样拿着机器人：【USB 口朝上、屏幕对着你】，手托住身体让两条腿能自由摆动（不要碰到桌面/手）。")
        say("   我会依次让 1 号、2 号舵机（摆腿）和 3 号（滑块）各动 4 秒，动作很小（约 12°）。")
        self.io.pause("   拿好了就继续。")
        obs = {}
        for sid in (1, 2):
            c = [0, 0, 0]
            c[sid - 1] = POSE_COUNTS
            say("   ▶ %d 号舵机 +%d 计数，看好……" % (sid, POSE_COUNTS))
            offs = self._pose(*c)
            if offs.get(sid) is not None and abs(offs[sid] - POSE_COUNTS) > 8:
                say("   ! 舵机只到了 %+d（目标 +%d）：腿可能被挡住了" % (offs[sid], POSE_COUNTS))
            side = self.io.choose("   %d 号舵机让【哪条腿】动了？（你看到的）" % sid, ["你左手边那条", "你右手边那条", "都没动 / 看不清"])
            if side == 3:
                raise Abort("看不清：请重新运行 setup / unclear, run setup again")
            direction = self.io.choose("   这条腿的脚往哪边摆？", ["朝你这一侧（屏幕方向）= 向前", "背离你 = 向后", "看不清"])
            if direction == 3:
                raise Abort("看不清：请重新运行 setup / unclear, run setup again")
            obs[sid] = (side, direction)
        say("   ▶ 3 号（滑块）+%d 计数，看好……" % POSE_COUNTS)
        self._pose(0, 0, POSE_COUNTS)
        lift = self.io.choose("   滑块让腿/身体怎么动了？", ["腿向上缩进身体（身体变矮）", "腿向下伸出（身体变高）", "看不清"])
        if lift == 3:
            raise Abort("看不清：请重新运行 setup / unclear, run setup again")

        if obs[1][0] == obs[2][0]:
            raise Abort("1 号和 2 号动的是同一边的腿？两个舵机应该各管一条腿。请检查后重新运行 / both servos moved the same leg")
        # viewer-right = the robot's LEFT arm (the robot faces the viewer)
        swap = 1 if obs[1][0] == 1 else 0          # ID1 moved the leg on the viewer's left = the robot's right arm -> swap
        s1 = 1 if obs[1][1] == 1 else -1           # hypothesis (a): ID1 +count = foot forward
        s2 = 1 if obs[2][1] == 2 else -1           # hypothesis (a): ID2 +count = foot backward
        s3 = 1 if lift == 1 else -1                # hypothesis (a): +count = legs retract
        say("   => 1号(+) 脚%s、2号(+) 脚%s、滑块(+) %s；1 号是机器人的%s臂" % (
            "向前" if obs[1][1] == 1 else "向后", "向前" if obs[2][1] == 1 else "向后", "收腿" if lift == 1 else "伸腿", "右" if swap else "左"))
        ok, lines = self.send("map %d %d %d %d" % (s1, s2, s3, swap), 6)
        if not ok:
            raise Abort("写入映射失败 / storing the mapping failed")
        say("   映射已保存。用自检姿势再确认一遍：")
        self.io.pause("   继续拿好机器人（腿自由悬空）。")
        self.send("selftest 1", 6, quiet=True)
        time.sleep(3 * TIME_SCALE)
        fwd = self.io.choose("   ▶ 两只脚是不是都【向前（朝你）】摆了约 20°？", ["是", "不是"])
        self.send("selftest 2", 6, quiet=True)
        time.sleep(3 * TIME_SCALE)
        up = self.io.choose("   ▶ 两条腿是不是都【向上缩】了约 12 mm？", ["是", "不是"])
        self.send("selftest 0", 6, quiet=True)
        if fwd != 1 or up != 1:
            raise Abort("自检不对：映射还有问题，请把这一步的现象告诉我 / self-test disagrees - tell me what you saw")
        self.results["map"] = (s1, s2, s3, swap)

    # ---------------------------------------------------------------------------------------------------------------
    def dry_run(self):
        say = self.io.say
        say("")
        say("== 3/3 试运行（不动舵机）/ dry run ==")
        ok, lines = self.send("learn dry", 8)
        if not ok:
            raise Abort("learn dry 失败 / learn dry failed")
        self.io.pause("   把机器人【竖直站立】放稳、不要动。我看 6 秒，应该一直是 STAND。")
        sts = []
        for l in self.watch(6.0):
            m = RE_ST.search(l)
            if m:
                sts.append(m.group(1))
        say("   状态：%s" % (", ".join(sorted(set(sts))) or "（没读到）"))
        stand_ok = bool(sts) and set(sts) == {"STAND"}
        if not sts:
            say("   ! 没读到状态：姿态滤波器没启动，保持静止再试")
        elif not stand_ok:
            say("   ! 竖直静止时出现了 %s：标定或姿态估计有问题" % ", ".join(sorted(set(sts) - {"STAND"})))
        if self.io.confirm("   要测一下摔倒检测吗？（我会让你把机器人仰面/趴着放倒 3 秒，再扶起来；舵机不会动）"):
            self.io.pause("   现在把机器人【仰面（背朝下）平放】，3 秒后我给结果。")
            seen = []
            for l in self.watch(5.0):
                m = RE_ST.search(l)
                if m:
                    seen.append(m.group(1))
            say("   平放时状态：%s" % (", ".join(dict.fromkeys(seen)) or "（没读到）"))
            if "FALLEN" not in seen:
                say("   ! 没有判定为摔倒：期望出现 FALLEN（再 GETUP1，因为是试运行所以不会真动）")
            else:
                say("   摔倒检测正常。")
        self.send("learn off", 6, quiet=True)
        self.results["dry"] = stand_ok

    # ---------------------------------------------------------------------------------------------------------------
    def run(self, stages=("pre", "imu", "map", "dry")):
        say = self.io.say
        try:
            if "pre" in stages:
                self.preflight()
            if "imu" in stages:
                self.calibrate_imu()
            if "map" in stages:
                self.servo_directions()
            if "dry" in stages:
                self.dry_run()
        except Abort as e:
            say("")
            say("✗ 中止：%s" % e)
            self.dev.send("estop")
            return 1
        except KeyboardInterrupt:
            self.dev.send("estop")
            say("\n^C -> 急停 / emergency stop")
            return 130
        say("")
        say("✓ 准备好了。结果：%s" % self.results)
        say("  下一步（机器人放在软垫上、手扶着）：")
        say("    python3 tools/tars_log/tars_serial.py learn on go 30        # 学习模式站立 30 秒后自动关闭（Ctrl-C 急停）")
        say("    python3 tools/tars_log/tars_serial.py test walk 20 6        # 学习模式开着时，走 6 秒并录数据")
        return 0
