"""The firmware supervisor (tars_ctrl.c with the v9 gaits and the learned get-up network, C code in the loop) driving the calibrated simulator,
with the possibility to take over the servos for a scripted segment (the way a 'motion' state of the real supervisor would).

CtlRun.ctl_step(vx, vyaw) runs the supervisor for one 20 ms step and returns its servo targets; CtlRun.apply(...) sends targets to the simulated servos and records
the frame.  A scripted segment calls apply() with its own targets (rate-limited by `Slew`) while the supervisor keeps running (its outputs are ignored): when
the script is over the supervisor simply takes over again.
"""
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fun_sim as fs                                   # noqa: E402
from fun_sim import DT, HIP, LIFT                      # noqa: E402
from c_in_loop import CController                      # noqa: E402
from test_c_loop import make_ctl                       # noqa: E402

ROOT = fs.ROOT
GAITS = dict(gait_json=os.path.join(ROOT, 'data/walk_cma_cal/fwd_s5.json'),
             back_json=os.path.join(ROOT, 'data/walk_cma_cal/back_s5_mirror_lift2_x085.json'),
             turn_json=os.path.join(ROOT, 'data/walk_cma_cal/turn_c3.json'),
             turnr_json=None)                            # right turn = mirror of the left turn gait (as in the firmware set v9)
KEY = os.path.join(ROOT, 'data/getup_cma/back_tau1.0_rate8.0.npy')
STATE_NAMES = CController.STATES
_CTL = {}


def get_ctl():
    if 'c' not in _CTL:
        _CTL['c'] = make_ctl(GAITS['gait_json'], KEY, back_json=GAITS['back_json'], turn_json=GAITS['turn_json'], turnr_json=GAITS['turnr_json'])
    return _CTL['c']


class Slew:
    """servo speed limits for a scripted segment (hips 8 rad/s, lift 0.25 m/s) starting from the targets in force"""

    def __init__(self, cur):
        self.cur = np.array(cur, float)

    def __call__(self, target):
        lim = np.array([fs.RATE_HIP * DT, fs.RATE_HIP * DT, fs.RATE_LIFT * DT])
        self.cur = self.cur + np.clip(np.asarray(target, float) - self.cur, -lim, lim)
        return self.cur.copy()


class CtlRun:
    def __init__(self, seed=0, randomize=False, T=30.0, getup_mode=1, straight=True):
        self.ctl = get_ctl()
        self.ctl.set_getup_mode(getup_mode)
        self.ctl.set_heading(trim=0.0, kp=1.5 if straight else 0.0, kd=0.0)
        self.env = fs.make_env(seed=seed, randomize=randomize, T=T)
        env = self.env
        self.fifo = deque(maxlen=64)
        self.ctl.reset(2.0)

        def hook(acc, gyr):
            self.fifo.append((np.array(acc), np.array(gyr)))
            lag = env.pr.get('imu_lag', 0) if hasattr(env, 'pr') else 0
            a_, g_ = self.fifo[-1 - min(lag, len(self.fifo) - 1)]
            self.ctl.imu(a_, g_, 0.01)
        env.imu_hook = hook
        env.reset()
        self.ctl.set_kp(env.filter.kp)
        self.lag = env.pr.get('imu_lag', 0)
        self.frames = []
        self.t = 0.0
        self.state = 0
        self.out = (0.0, 0.0, 0.0)
        self.stats = dict(max_tilt=0.0, falls=0, states=set())
        self._prev_state = 0
        self.slew = None

    def ctl_step(self, vx=0.0, vyaw=0.0):
        g_ = self.fifo[-1 - min(self.lag, len(self.fifo) - 1)][1]
        ql, qr, z, st = self.ctl.step(g_, DT, vx, vyaw)
        self.state = st
        self.out = (ql, qr, z)
        return ql, qr, z, st

    def apply(self, ql, qr, z, label=None, extra=None, push=None):
        """send servo targets [rad, rad, m]; push = (Fx, Fy) world force [N] on the torso for this step"""
        env = self.env
        if push is not None:
            env.push_force = np.array([push[0], push[1], 0.0])
            env.push_until = env.t + DT - 1e-6
        env.step(np.array([ql / HIP, qr / HIP, z / LIFT]))
        d = env.d
        tilt = fs.torso_tilt_deg(env)
        s = self.stats
        s['max_tilt'] = max(s['max_tilt'], tilt)
        if self.state == 5 and self._prev_state != 5:
            s['falls'] += 1
        self._prev_state = self.state
        s['states'].add(STATE_NAMES[self.state])
        self.frames.append((self.t, d.qpos.copy(), label if label is not None else STATE_NAMES[self.state], extra))
        self.t += DT

    def step(self, vx=0.0, vyaw=0.0, label=None, extra=None, push=None):
        ql, qr, z, st = self.ctl_step(vx, vyaw)
        self.apply(ql, qr, z, label, extra, push)
        return st

    def script(self, fn, T, label='', extra_fn=None):
        """take over the servos for T seconds: fn(t) -> (hl, hr, z); starts from the targets the supervisor last commanded"""
        sl = Slew(self.out)
        n = int(round(T / DT))
        for i in range(n):
            self.ctl_step(0.0, 0.0)                    # the supervisor keeps running (its outputs are ignored)
            q = sl(fn(i * DT))
            self.out = tuple(q)
            self.apply(q[0], q[1], q[2], label, extra_fn(i * DT) if extra_fn else None)

    def upright(self):
        return fs.torso_tilt_deg(self.env) < 10.0
