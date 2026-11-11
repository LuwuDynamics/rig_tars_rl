"""A small MuJoCo harness for choreographed motions of the Tars robot: the three servo targets (left hip, right hip, lift) as functions of time,
played through the same servo model, command delay, count quantisation and speed limits as the controllers (calibrated simulator).

Conventions (as everywhere in this project): hip q > 0 = the foot end of the arm bar swings FORWARD [rad]; lift z > 0 = the arms retract UP relative to the torso [m]
(z < 0 = the arms push down: the robot stands tall on its arms).  The arm bars pivot about their MIDDLE (hip axis at mid-height of the torso).

Two basic postures follow from the geometry (see FUN_MOTIONS.md):
  READY  z = +14 mm : the arms are retracted and clear of the floor, the robot stands on its torso base -> the arms can swing freely (wave, cheer, windmill ...)
  TALL   z = -14 mm : the arms stand on the floor like legs, the torso hangs on the hip axis 14 mm above the floor and can pitch freely there:
                      with both hips commanded alike the torso pitches by exactly q (q > 0 = bow forward / nose down, q < 0 = look up) as long as the feet do not slip
                      (bow, nod, head-bang, sneeze ...).  The torso's centre of mass is 14 mm above the hip axis, so the hips hold it against a gravity torque of at most
                      0.025 N m (about 10 % of what they can give), and the robot's centre of mass moves less than 7 mm fore/aft at any angle -> it stays over its feet.
"""
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('sim', 'rl', 'calib'):
    sys.path.insert(0, os.path.join(ROOT, p))
from tars_env import TarsEnv, DT                     # noqa: E402
import eval_cal                                      # noqa: E402

HIP = 2.4
LIFT = 0.014
RATE_HIP = 8.0           # rad/s: the servo speed limit used by the firmware supervisor
RATE_LIFT = 0.25         # m/s: about the speed of the lift servo itself (0.27 m/s in the model)
READY = 0.014            # lift target [m]: arms retracted, robot on its torso base
TALL = -0.014            # lift target [m]: arms extended, robot on its arms, torso free to pitch


def make_env(seed=0, randomize=False, calibrated=True, T=12.0, **kw):
    cfg = dict(task='walk', hip_scale=HIP, lift_scale=LIFT, max_steps=int(T / DT), randomize=randomize, difficulty=1.0, cmd_fixed=(0, 0),
               no_terminate=True, push_prob_per_s=0.0, obs_q_mode='cmd')
    if calibrated:
        c = eval_cal.calibrated_kwargs(lag=1, lag_jitter=1)
        c.pop('ctrl_dt', None)
        cfg.update(c)
    cfg.update(kw)
    return TarsEnv(seed=seed, **cfg)


def smooth(x):
    """0 -> 1 with zero slope at both ends"""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    return smooth((t - t0) / max(1e-6, t1 - t0))


def kf(t, pts):
    """smooth interpolation through (time, value) keyframes (flat before the first and after the last)"""
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t < t1:
            return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
    return pts[-1][1]


def torso_pitch_deg(env):
    """torso pitch, + = nose down (the front of the torso, the screen side, tips down)"""
    R = env.d.xmat[env.id_torso].reshape(3, 3)
    return math.degrees(math.atan2(-R[2, 0], R[2, 2]))


def torso_tilt_deg(env):
    R = env.d.xmat[env.id_torso].reshape(3, 3)
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


class Player:
    """plays targets(t) -> (hl, hr, z) [rad, rad, m] with the servo speed limits; records the free-joint pose of every step and a few statistics"""

    def __init__(self, env, rate_hip=RATE_HIP, rate_lift=RATE_LIFT):
        self.env = env
        self.rate_hip, self.rate_lift = rate_hip, rate_lift
        self.env.reset()
        d = env.d
        self.cur = np.array([d.qpos[env.qadr['hip_l']], d.qpos[env.qadr['hip_r']], d.qpos[env.qadr['lift']]])
        self.frames = []                      # (t, qpos, label, extra)
        self.stats = dict(max_tilt=0.0, max_pitch=-1e9, min_pitch=1e9, min_z=1e9, max_hip_speed=0.0, hip_sat=0.0, hip_rms=0.0, steps=0, min_gap=1e9)
        self.t = 0.0
        self._fromto = np.zeros(6)
        self._sat = 0
        self._sq = 0.0

    def step(self, target, label='', extra=None):
        env = self.env
        tgt = np.asarray(target, float)
        lim = np.array([self.rate_hip * DT, self.rate_hip * DT, self.rate_lift * DT])
        self.cur = self.cur + np.clip(tgt - self.cur, -lim, lim)
        a = np.clip(self.cur / np.array([HIP, HIP, LIFT]), -1.0, 1.0)
        env.step(a)
        self.record(label, extra)

    def record(self, label='', extra=None):
        env, d, s = self.env, self.env.d, self.stats
        s['max_tilt'] = max(s['max_tilt'], torso_tilt_deg(env))
        p = torso_pitch_deg(env)
        s['max_pitch'] = max(s['max_pitch'], p)
        s['min_pitch'] = min(s['min_pitch'], p)
        s['min_z'] = min(s['min_z'], float(d.qpos[2]))
        if abs(p) > 3.0 and self.t > 0.7:                 # clearance between the torso box and the floor while the torso is pitched (the arms stand on the floor, TALL posture; not during the first 0.7 s: the lift is still extending)
            g = mujoco.mj_geomDistance(env.m, d, env.g_boxes[0], env.g_floor, 0.2, self._fromto)
            s['min_gap'] = min(s['min_gap'], float(g))
        s['max_hip_speed'] = max(s['max_hip_speed'], abs(d.qvel[7]), abs(d.qvel[8]))
        lim = env.m.actuator_forcerange[:2, 1]
        f = np.abs(d.actuator_force[:2])
        self._sat += int(np.any(f >= 0.98 * lim))
        self._sq += float(np.sum((f / env.tau_nom[:2]) ** 2))
        s['steps'] += 1
        s['hip_sat'] = self._sat / s['steps']
        s['hip_rms'] = math.sqrt(self._sq / (2 * s['steps']))
        self.frames.append((self.t, d.qpos.copy(), label, extra))
        self.t += DT

    def run(self, fn, T, label=''):
        n = int(round(T / DT))
        t0 = self.t
        for i in range(n):
            self.step(fn(self.t - t0), label)
        return self
