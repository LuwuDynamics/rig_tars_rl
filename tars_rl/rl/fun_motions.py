"""Fun motions for the Tars robot in MuJoCo (calibrated simulator, same servo model / delay / speed limits as the controllers).

Two postures make most of them possible (see fun_sim.py):  READY (arms retracted, robot on its torso base -> the arms are free: wave, cheer, windmill, robot dance,
swim, twist) and TALL (robot on its arms, torso hinged on the hip axis -> the torso pitches freely: bow, nod, head-bang, sneeze, look-up, fold).  Three more clips use
the firmware supervisor (tars_ctrl.c, C code in the loop) for get-up and walking: play-dead, hello, spin.  `medley` chains six of the others.

usage:  python rl/fun_motions.py list
        python rl/fun_motions.py check [names] [--n 20]       stability under the randomised plant (friction, mass, servo strength, delays, IMU lag)
        python rl/fun_motions.py strips [names]                key-frame strips into /tmp/fun/ (quick look while tuning)
        python rl/fun_motions.py render [names]                MP4 + GIF + strip into media/fun/
        python rl/fun_motions.py all [--n 30]                  the whole package: the 10 + 6 clips, overview.png, index.html, summary.json (what FUN_MOTIONS.md quotes)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fun_sim import *                                  # noqa: E402,F401,F403
import fun_sim as fs                                   # noqa: E402

OUT = os.path.join(fs.ROOT, 'media', 'fun')
TWO_PI = 2 * math.pi
MOTIONS = {}                                           # name -> spec


def motion(name, title, sub, T, family, view=None, tilt_limit=None, face=None):
    def deco(fn):
        MOTIONS[name] = dict(name=name, title=title, subtitle=sub, T=T, family=family, fn=fn, view=view or {}, tilt_limit=tilt_limit, face=face)
        return fn
    return deco


FRONT = dict(azim=140, elev=-12, dist=0.44, lookat_z=0.07)
SIDE = dict(azim=150, elev=-12, dist=0.44, lookat_z=0.075)


# ================================================================================================ READY family: the arms swing freely
@motion('wave', '招手', 'wave hello  -  arms retracted, right arm up and waving', 6.6, 'ready', FRONT, tilt_limit=6.0)
def wave(t):
    z = kf(t, [(0, 0), (0.6, READY), (5.9, READY), (6.5, 0)])
    amp = ramp(t, 1.5, 1.9) * (1 - ramp(t, 4.4, 4.8))
    q = kf(t, [(0.6, 0), (1.5, 1.25), (4.4, 1.25), (5.3, 0)]) + 0.35 * amp * math.sin(TWO_PI * 2.2 * (t - 1.5))
    return (0.0, q, z)


def wave_fn(waves=2.0, hand=1):
    """a short wave (also used as a scripted segment after walking): arm up, `waves` seconds of waving, arm down"""
    t_up = 1.1
    t_we = t_up + 0.4 + waves
    t_dn = t_we + 0.9
    t_end = t_dn + 0.6

    def fn(t):
        z = kf(t, [(0, 0), (0.5, READY), (t_dn + 0.1, READY), (t_end, 0)])
        amp = ramp(t, t_up + 0.1, t_up + 0.4) * (1 - ramp(t, t_we - 0.3, t_we))
        q = kf(t, [(0.5, 0), (t_up, 1.25), (t_we, 1.25), (t_dn, 0)]) + 0.35 * amp * math.sin(TWO_PI * 2.2 * (t - t_up - 0.1))
        return (0.0, q, z) if hand == 1 else (q, 0.0, z)
    fn.T = t_end + 0.1
    return fn


wave_short = wave_fn(2.2)


def cheer_fn(pumps=3):
    t_up = 1.3                                           # arms are up
    t_pe = t_up + 0.8 * pumps                            # end of the pumps
    t_se = t_pe + 1.0                                    # end of the shiver
    t_dn = t_se + 0.9                                    # arms are down
    t_end = t_dn + 0.7

    def fn(t):
        z = kf(t, [(0, 0), (0.6, READY), (t_dn + 0.1, READY), (t_end, 0)])
        base = kf(t, [(0.6, 0), (t_up, 1.45), (t_se, 1.45), (t_dn, 0)])
        pump = 0.0
        if t_up <= t < t_pe:                             # big pumps: down to 0.65 and back up (both arms together: slow enough not to rock the torso)
            pump = -0.8 * math.sin(math.pi * (t - t_up) / 0.8) ** 2
        shiver = 0.15 * ramp(t, t_pe + 0.2, t_pe + 0.4) * (1 - ramp(t, t_se - 0.3, t_se - 0.1)) * math.sin(TWO_PI * 4.0 * (t - t_pe - 0.2))   # against each other: the reaction on the torso cancels
        return (base + pump + shiver, base + pump - shiver, z)
    fn.T = t_end + 0.1
    return fn


_cheer3 = cheer_fn(3)
cheer = motion('cheer', '欢呼', 'hooray  -  both arms up, three big pumps, then a happy shiver', _cheer3.T, 'ready', FRONT, tilt_limit=6.0)(_cheer3)
cheer2 = cheer_fn(2)                                     # the short version that follows the get-up in play-dead and in the medley


@motion('windmill', '风车', 'windmill  -  the two arms sweep through +-100 deg in opposite directions', 6.4, 'ready', FRONT, tilt_limit=6.0)
def windmill(t):
    z = kf(t, [(0, 0), (0.6, READY), (5.7, READY), (6.3, 0)])
    a = 1.7 * ramp(t, 0.6, 1.6) * (1 - ramp(t, 4.6, 5.6))
    s = math.sin(TWO_PI * 0.5 * (t - 0.6))
    return (a * s, -a * s, z)


def _pose_fn(poses, t0=0.6, hold=0.36, snap=0.24, snap_inphase=0.4):
    """staccato choreography: (hl, hr) poses, each reached with a smooth snap and held; the arms moving together get a slower snap (they would rock the torso)"""
    times = []
    t = t0
    for k, p in enumerate(poses):
        prev = poses[k - 1] if k else (0.0, 0.0)
        dl, dr = p[0] - prev[0], p[1] - prev[1]
        together = dl * dr > 0.05
        sn = snap_inphase if together and max(abs(dl), abs(dr)) > 0.8 else snap
        times.append((t, t + sn, p))
        t += sn + hold
    return times, t


_RD_POSES = [(1.55, 0.0), (0.0, 1.55), (1.55, 1.55), (0.0, 0.0), (1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (2.1, -2.1), (0.0, 0.0), (0.9, 1.55), (1.55, 0.9), (0.9, 1.55), (0.0, 0.0)]
_RD_TIMES, _RD_END = _pose_fn(_RD_POSES)


@motion('robotdance', '机器人舞', 'the robot dance  -  sharp snaps between held arm poses, on the beat', _RD_END + 1.2, 'ready', FRONT, tilt_limit=6.0)
def robotdance(t):
    z = kf(t, [(0, 0), (0.6, READY), (_RD_END + 0.1, READY), (_RD_END + 0.7, 0)])
    hl = hr = 0.0
    for k, (ta, tb, p) in enumerate(_RD_TIMES):
        prev = _RD_POSES[k - 1] if k else (0.0, 0.0)
        if t >= ta:
            u = smooth((t - ta) / (tb - ta))
            hl = prev[0] + (p[0] - prev[0]) * u
            hr = prev[1] + (p[1] - prev[1]) * u
    return (hl, hr, z)


@motion('swim', '游泳', 'freestyle  -  arms stroke in turn while the shoulders (the lift) circle', 6.6, 'ready', FRONT, tilt_limit=8.0)
def swim(t):
    a = ramp(t, 0.6, 1.4) * (1 - ramp(t, 5.0, 5.8))
    ph = TWO_PI * 0.7 * (t - 0.6)
    q = 1.2 * math.sin(ph)
    z = kf(t, [(0, 0), (0.6, READY), (5.9, READY), (6.5, 0)]) - 0.005 * a * (1 - math.cos(2 * ph)) / 2
    return (a * q, -a * q, z)


@motion('twist', '扭一扭', 'twist  -  the arms swing against each other fast and small', 5.6, 'ready', FRONT, tilt_limit=6.0)
def twist(t):
    z = kf(t, [(0, 0), (0.6, READY), (4.9, READY), (5.5, 0)])
    a = 0.6 * ramp(t, 0.6, 1.2) * (1 - ramp(t, 4.2, 4.8))
    s = math.sin(TWO_PI * 1.6 * (t - 0.6))
    return (a * s, -a * s, z)


# ================================================================================================ TALL family: the torso pitches on the hips
@motion('bow', '鞠躬', 'bow  -  a slow deep bow (55 deg), a short pause, then a quick second one', 8.0, 'tall', SIDE, tilt_limit=62.0)
def bow(t):
    z = kf(t, [(0, 0), (0.6, TALL), (7.3, TALL), (7.9, 0)])
    q = kf(t, [(0.6, 0), (2.2, 0.95), (3.4, 0.95), (4.8, 0.0), (5.2, 0.0), (5.7, 0.5), (6.0, 0.5), (6.5, 0.0)])
    return (q, q, z)


@motion('nod', '点头', 'nod yes  -  three nods', 5.6, 'tall', SIDE, tilt_limit=26.0)
def nod(t):
    z = kf(t, [(0, 0), (0.6, TALL), (4.9, TALL), (5.5, 0)])
    a = ramp(t, 0.6, 1.0) * (1 - ramp(t, 4.0, 4.4))
    q = a * (-0.10 + 0.40 * (1 - math.cos(TWO_PI * 1.5 * (t - 0.9))) / 2)
    return (q, q, z)


@motion('headbang', '甩头', 'head-bang  -  big fast nods (1.8 Hz) between 9 deg up and 43 deg down', 7.0, 'tall', SIDE, tilt_limit=62.0)
def headbang(t):
    z = kf(t, [(0, 0), (0.6, TALL), (6.3, TALL), (6.9, 0)])
    a = ramp(t, 0.6, 1.4) * (1 - ramp(t, 5.4, 6.2))
    q = a * (0.30 + 0.45 * math.sin(TWO_PI * 1.8 * (t - 0.9) - math.pi / 2))
    return (q, q, z)


@motion('sneeze', '打喷嚏', 'a-choo!  -  lean back and tremble, then snap forward and ring out', 7.0, 'tall', SIDE, tilt_limit=62.0)
def sneeze(t):
    z = kf(t, [(0, 0), (0.6, TALL), (6.4, TALL), (7.0, 0)])
    q = kf(t, [(0.6, 0.0), (2.3, -0.45)])
    tremble = 0.05 * ramp(t, 1.6, 2.3) * (1 - ramp(t, 2.9, 3.0)) * math.sin(TWO_PI * 7.0 * t)
    if t < 3.0:
        q += tremble
    else:                                                # snap forward (3.0 s), hold, then ring out
        q = kf(t, [(3.0, -0.45), (3.2, 0.65), (3.55, 0.65)])
        if t > 3.55:
            tt = t - 3.55
            q = 0.65 * math.exp(-2.6 * tt) * math.cos(TWO_PI * 1.8 * tt)
    return (q, q, z)


@motion('lookup', '仰望星空', 'stargazing  -  leans back and looks up (63 deg), then slowly sways', 8.0, 'tall', SIDE, tilt_limit=76.0)
def lookup(t):
    z = kf(t, [(0, 0), (0.6, TALL), (7.3, TALL), (7.9, 0)])
    q = kf(t, [(0.6, 0.0), (2.6, -1.1)])
    sway = 0.12 * ramp(t, 3.4, 4.4) * (1 - ramp(t, 5.4, 6.0)) * math.sin(TWO_PI * 0.7 * (t - 3.4))
    back = ramp(t, 5.4, 7.0)
    return (q * (1 - back) + sway, q * (1 - back) + sway, z)


@motion('fold', '弯腰到底', 'touch-the-floor fold  -  bends over to 115 deg and looks back between its legs', 8.0, 'tall', SIDE, tilt_limit=125.0)
def fold(t):
    z = kf(t, [(0, 0), (0.6, TALL), (7.3, TALL), (7.9, 0)])
    q = kf(t, [(0.6, 0.0), (3.2, 2.0), (4.8, 2.0), (6.9, 0.0)])
    return (q, q, z)


@motion('rockchair', '摇椅', 'rocking chair  -  slow pitching +-35 deg', 8.0, 'tall', SIDE, tilt_limit=45.0)
def rockchair(t):
    z = kf(t, [(0, 0), (0.6, TALL), (7.3, TALL), (7.9, 0)])
    a = ramp(t, 0.6, 1.8) * (1 - ramp(t, 5.8, 7.0))
    q = a * 0.6 * math.sin(TWO_PI * 0.5 * (t - 0.6))
    return (q, q, z)


@motion('squat', '深蹲', 'squats  -  the legs (arms) push the body up and let it down, four times', 6.0, 'tall', SIDE, tilt_limit=6.0)
def squat(t):
    n = ramp(t, 0.3, 0.9) * (1 - ramp(t, 4.6, 5.2))
    return (0.0, 0.0, 0.014 * n * math.cos(TWO_PI * 0.9 * (t - 0.9)))



# ================================================================================================ EMOTIONS (TALL posture): posture, height, tempo and rhythm of the torso on its hips
# What carries an emotion on this body: leaning back = open / proud / startled, leaning forward = closed / sad / shy, the body low = sad / scared / tired and tall = alert /
# proud / surprised, jerky and fast = angry / scared, slow and smooth = sad / sleepy / loving, a rhythm = joy (bounce), laughter (heave), fear (tremble), sobbing (shudder).
# All of them stand on the arms (the posture that was tried on the real robot), keep the torso at least ~5 mm above the floor while it is pitched, and end in the stand pose.
# `face` = the emoji of the screen (main/boards/tars/emoji/*.eaf) that goes with it.
def _lift(t, t_end, *pts):
    """lift target for an emotion: TALL at 0.6 s, the keyframes `pts` (time, value) in between, TALL again at t_end - 0.6 s, the stand pose (0) at t_end"""
    return kf(t, [(0, 0), (0.6, TALL)] + list(pts) + [(t_end - 0.6, TALL), (t_end, 0)])


def _bump(t, t0, t1, t2, t3):
    """0 -> 1 between t0 and t1, 1 until t2, back to 0 at t3 (smooth)"""
    return ramp(t, t0, t1) * (1 - ramp(t, t2, t3))


@motion('happy', '开心', 'happy  -  bounces for joy, then a proud ta-da', 6.6, 'emotion', SIDE, tilt_limit=45.0, face='happy')
def happy(t):
    a = _bump(t, 1.0, 1.4, 4.0, 4.4)
    ph = TWO_PI * 2.2 * (t - 1.0)
    bounce = 0.5 * (1 - math.cos(ph))                                  # 1 = the lowest point of a bounce
    z = _lift(t, 6.6) + 0.006 * a * bounce                             # the body dips 6 mm and springs back
    lean = kf(t, [(0.6, 0.0), (1.0, -0.10), (4.0, -0.10), (4.6, -0.28), (5.5, -0.28), (6.0, 0.0)])
    return ((lean + 0.12 * a * (2 * bounce - 1),) * 2) + (z,)


@motion('sad', '难过', 'sad  -  droops slowly, shudders with sobs, then slowly picks itself up', 10.4, 'emotion', SIDE, tilt_limit=60.0, face='sad')
def sad(t):
    z = _lift(t, 10.4, (1.0, TALL), (3.8, -0.009), (7.4, -0.009), (9.4, TALL))
    base = kf(t, [(0.6, 0.0), (1.0, 0.0), (3.8, 0.50), (7.4, 0.50), (9.4, 0.0)])
    sob = 0.05 * (_bump(t, 4.2, 4.4, 5.4, 5.6) + _bump(t, 6.0, 6.2, 7.0, 7.2)) * math.sin(TWO_PI * 2.0 * (t - 4.2))
    q = base + sob
    return (q, q, z + 0.001 * (_bump(t, 4.2, 4.4, 5.4, 5.6) + _bump(t, 6.0, 6.2, 7.0, 7.2)) * math.sin(TWO_PI * 2.0 * (t - 4.2) + 1.0))


@motion('angry', '生气', 'angry  -  tense and trembling, then three sharp lunges', 7.4, 'emotion', SIDE, tilt_limit=60.0, face='angry')
def angry(t):
    base = kf(t, [(0.6, 0.0), (1.1, 0.0), (1.3, 0.20), (6.1, 0.20), (7.0, 0.0)])
    seethe = 0.025 * _bump(t, 1.5, 1.8, 2.2, 2.4) * math.sin(TWO_PI * 9.0 * t)
    lunge = stomp = 0.0
    for tl, amp in ((2.5, 0.28), (3.6, 0.34), (4.7, 0.40)):            # out fast, a short hold, back
        u = t - tl
        if 0.0 <= u < 0.7:
            s_ = smooth(u / 0.18) if u < 0.18 else (1.0 if u < 0.30 else 1.0 - smooth((u - 0.30) / 0.40))
            lunge += amp * s_
            stomp += 0.005 * s_
    q = base + seethe + lunge
    return (q, q, _lift(t, 7.4) + stomp)


@motion('scared', '害怕', 'scared  -  flinches back and crouches, trembles, peeks, slowly calms down', 7.4, 'emotion', SIDE, tilt_limit=45.0, face='shocked')
def scared(t):
    base = kf(t, [(0.6, 0.0), (1.2, 0.0), (1.38, -0.27), (3.6, -0.27), (4.0, -0.18), (4.4, -0.27), (5.2, -0.27), (6.6, 0.0)])
    crouch = kf(t, [(0.6, 0.0), (1.2, 0.0), (1.38, 0.003), (5.2, 0.003), (6.6, 0.0)])
    trem = _bump(t, 1.6, 1.9, 4.6, 5.2)
    q = base + 0.04 * trem * math.sin(TWO_PI * 8.0 * t)
    return (q, q, _lift(t, 7.4) + crouch + 0.001 * trem * math.sin(TWO_PI * 8.0 * t + 1.0))


@motion('surprised', '惊讶', 'surprised  -  a sudden pop up and back, frozen, a second jolt, then it settles', 5.2, 'emotion', SIDE, tilt_limit=45.0, face='surprised')
def surprised(t):
    jolt = -0.07 * math.sin(math.pi * min(1.0, max(0.0, (t - 2.5) / 0.35))) ** 2
    q = kf(t, [(0.6, 0.0), (1.2, 0.10), (1.6, 0.10), (1.76, -0.40), (3.0, -0.40), (4.4, 0.0)]) + jolt
    z = _lift(t, 5.2, (1.0, -0.010), (1.6, -0.010), (1.76, TALL))
    return (q, q, z)


@motion('shy', '害羞', 'shy  -  looks down and fidgets, peeks up at you, looks down again', 9.4, 'emotion', SIDE, tilt_limit=55.0, face='embarrassed')
def shy(t):
    q = kf(t, [(0.6, 0.0), (1.0, 0.0), (2.4, 0.38), (4.0, 0.38), (4.4, 0.05), (5.0, 0.05), (5.4, 0.44), (7.4, 0.44), (8.6, 0.0)])
    q += 0.035 * (_bump(t, 2.6, 3.0, 3.6, 4.0) + _bump(t, 5.8, 6.2, 6.8, 7.2)) * math.sin(TWO_PI * 0.8 * (t - 2.6))
    z = _lift(t, 9.4, (1.0, TALL), (2.4, -0.010), (7.6, -0.010), (8.6, TALL))
    return (q, q, z)


@motion('curious', '好奇', 'curious  -  leans in, sniffs, leans back to think, leans in even closer', 8.4, 'emotion', SIDE, tilt_limit=50.0, face='thinking')
def curious(t):
    q = kf(t, [(0.6, 0.0), (1.0, 0.0), (2.2, 0.30), (3.4, 0.30), (4.2, -0.05), (4.7, -0.05), (5.9, 0.40), (6.8, 0.40), (7.8, 0.0)])
    q += 0.06 * _bump(t, 2.3, 2.5, 3.0, 3.3) * math.sin(TWO_PI * 3.0 * (t - 2.3))                    # sniff, sniff
    z = _lift(t, 8.4, (1.0, TALL), (5.9, -0.011), (6.8, -0.011), (7.6, TALL))
    return (q, q, z)


@motion('sleepy', '困', 'sleepy  -  nods off, jerks awake, nods off again, then a big yawn', 12.4, 'emotion', SIDE, tilt_limit=55.0, face='sleepy')
def sleepy(t):
    q = kf(t, [(0.6, 0.0), (1.0, 0.0), (3.4, 0.46), (3.65, 0.06), (4.2, 0.06), (6.6, 0.50), (6.85, 0.14), (7.4, 0.14), (9.0, -0.40), (10.0, -0.40), (11.6, 0.0)])
    z = _lift(t, 12.4, (1.0, TALL), (3.4, -0.010), (3.65, -0.013), (4.2, -0.013), (6.6, -0.009), (6.85, -0.012), (7.4, -0.012), (9.0, TALL), (10.0, TALL), (11.6, TALL))
    return (q, q, z)


@motion('proud', '得意', 'proud  -  chest out, swaggers, then one firm nod', 8.0, 'emotion', SIDE, tilt_limit=45.0, face='confident')
def proud(t):
    a = _bump(t, 2.6, 3.2, 5.2, 5.8)
    ph = TWO_PI * 0.7 * (t - 2.6)
    q = kf(t, [(0.6, 0.0), (1.0, 0.0), (2.4, -0.22), (5.4, -0.22), (5.75, 0.16), (6.35, -0.10), (7.2, 0.0)]) + 0.05 * a * math.sin(ph)
    z = _lift(t, 8.0) + 0.003 * a * 0.5 * (1 - math.cos(2 * ph))
    return (q, q, z)


@motion('laugh', '大笑', 'laughing  -  head thrown back, heaving with laughter, calming down', 7.2, 'emotion', SIDE, tilt_limit=45.0, face='laughing')
def laugh(t):
    a = _bump(t, 1.5, 2.0, 4.4, 5.4)
    ph = TWO_PI * 3.6 * (t - 1.5)
    q = kf(t, [(0.6, 0.0), (1.0, 0.0), (1.6, -0.18), (4.8, -0.18), (5.8, 0.0)]) + 0.14 * a * math.sin(ph)
    z = _lift(t, 7.2) + 0.004 * a * 0.5 * (1 - math.cos(ph + math.pi / 2))
    return (q, q, z)


@motion('loving', '撒娇', 'affectionate  -  leans in and rocks slowly, then nuzzles close', 9.0, 'emotion', SIDE, tilt_limit=55.0, face='loving')
def loving(t):
    a = _bump(t, 1.8, 2.8, 5.8, 6.6)
    base = kf(t, [(0.6, 0.0), (1.0, 0.0), (1.8, 0.16), (6.2, 0.16), (7.0, 0.40), (7.9, 0.40), (8.4, 0.0)])
    q = base + 0.17 * a * math.sin(TWO_PI * 0.55 * (t - 1.8))
    z = _lift(t, 9.0, (1.0, TALL), (7.0, -0.011), (7.9, -0.011), (8.2, TALL))
    return (q, q, z)


@motion('impatient', '不耐烦', 'impatient  -  taps its foot, then a long sigh', 7.6, 'emotion', SIDE, tilt_limit=45.0, face='confused')
def impatient(t):
    tap = _bump(t, 1.2, 1.4, 3.6, 3.8)
    ph = TWO_PI * 2.6 * (t - 1.2)
    q = kf(t, [(0.6, 0.0), (1.0, 0.10), (3.8, 0.10), (4.5, -0.25), (5.4, -0.25), (6.3, 0.0)]) + 0.03 * tap * math.sin(ph)
    z = _lift(t, 7.6) + 0.006 * tap * 0.5 * (1 - math.cos(ph))
    return (q, q, z)


EMOTIONS = ['happy', 'sad', 'angry', 'scared', 'surprised', 'shy', 'curious', 'sleepy', 'proud', 'laugh']          # the ten main ones; loving and impatient are extras


# ================================================================================================ medley: the motions chained
_MEDLEY = [('robotdance', 0.0, _RD_END + 0.2), ('wave', 0.6, 5.4), ('cheer2', 0.6, 5.2), ('bow', 0.0, 5.0), ('sneeze', 0.6, 5.7), ('headbang', 0.6, 6.9)]
_MEDLEY_FN = dict(robotdance=robotdance, wave=wave, cheer2=cheer2, bow=bow, sneeze=sneeze, headbang=headbang)
_MEDLEY_T = sum(b - a for _, a, b in _MEDLEY)


@motion('medley', '串烧', 'all in one go: robot dance, wave, cheer, bow, a-choo, head-bang', _MEDLEY_T, 'mixed', FRONT, tilt_limit=62.0)
def medley(t):
    t0 = 0.0
    for nm, a, b in _MEDLEY:
        if t < t0 + (b - a):
            return _MEDLEY_FN[nm](a + t - t0)
        t0 += b - a
    return _MEDLEY_FN['headbang'](_MEDLEY[-1][2])


# ================================================================================================ supervisor in the loop (C code): get-up, walking
HYBRIDS = {}


def hybrid(name, title, sub, view=None, speed=1.0):
    def deco(fn):
        HYBRIDS[name] = dict(name=name, title=title, subtitle=sub, runner=fn, view=view or {}, family='ctl', speed=speed)
        return fn
    return deco


def _finger(t, t_poke, p_contact, fwd):
    """a poking finger (decoration only: the push itself is a force on the torso) approaching from the front, pushing, withdrawing"""
    if t < t_poke - 0.55 or t > t_poke + 0.75:
        return None
    if t < t_poke:
        d = 0.11 * (1 - ramp(t, t_poke - 0.5, t_poke))
    elif t < t_poke + 0.12:
        d = -0.012 * ramp(t, t_poke, t_poke + 0.1)
    else:
        d = -0.012 + 0.14 * ramp(t, t_poke + 0.12, t_poke + 0.7)
    tip = p_contact + fwd * d
    back = tip + fwd * 0.085 + np.array([0, 0, 0.014])
    return {'deco': [('capsule', tip, back, 0.0075, (0.96, 0.76, 0.62, 1.0)), ('sphere', tip, 0.0085, (0.96, 0.76, 0.62, 1.0))]}


@hybrid('playdead', '装死', 'poked over, lies still, then stands up by itself (learned get-up network) and cheers', dict(azim=145, elev=-18, dist=0.55, lookat_z=0.06))
def run_playdead(seed=0, randomize=False, poke_force=8.0, after=None, getup_mode=1):
    import fun_ctl
    after = after or cheer2
    r = fun_ctl.CtlRun(seed=seed, randomize=randomize, T=40.0, getup_mode=getup_mode)
    env = r.env
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    fwd = R0[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    pos0 = env.d.xpos[env.id_torso].copy()
    p_contact = np.array([pos0[0], pos0[1], 0.118]) + fwd * 0.0235
    t_poke = 1.6
    fell, t_stand = False, None
    for i in range(int(25.0 / DT)):
        t = r.t
        push = None
        if abs(t - (t_poke + 0.04)) < DT / 2:
            push = (-fwd[0] * poke_force, -fwd[1] * poke_force)          # the finger pushes the robot backwards
        st = r.step(0.0, 0.0, extra=_finger(t, t_poke, p_contact, fwd), push=push)
        if st == 5:
            fell = True
        if fell and st == 0 and r.upright():
            t_stand = t_stand or t
            if t - t_stand > 1.2:
                break
        else:
            t_stand = None
    ok_up = fell and t_stand is not None
    if ok_up:
        r.script(after, after.T, 'cheer')
        for _ in range(int(0.8 / DT)):
            r.step(0.0, 0.0, label='STAND')
    s = dict(r.stats)
    s['states'] = sorted(s['states'])
    s.update(ok=bool(ok_up and r.upright()), final_tilt=fs.torso_tilt_deg(env), fell=fell, t_getup=(t_stand if t_stand else None))
    return dict(frames=r.frames, env=env, stats=s, T=r.frames[-1][0])


def _yaw(env):
    R = env.d.xmat[env.id_torso].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def _walk_for(r, T, vx=0.0, vyaw=0.0, label=None):
    for _ in range(int(round(T / DT))):
        r.step(vx, vyaw, label=label)


def _turn_by(r, angle, vyaw=20.0, tmax=20.0):
    """turn in place until the heading has changed by `angle` [rad] (+ = left); returns the time taken"""
    yaw_prev, acc, n = _yaw(r.env), 0.0, 0
    sign = 1.0 if angle > 0 else -1.0
    while abs(acc) < abs(angle) - 0.12 and n * DT < tmax:          # 0.12 rad: the gait keeps turning a little while it ramps down
        r.step(0.0, sign * vyaw)
        y = _yaw(r.env)
        acc += (y - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = y
        n += 1
    return n * DT


@hybrid('hello', '打招呼再见', 'walks up, waves hello, turns around (180 deg) and walks off  (x1.5)', dict(azim=145, elev=-22, dist=0.62, lookat_z=0.05, follow=False), speed=1.5)
def run_hello(seed=0, randomize=False):
    import fun_ctl
    r = fun_ctl.CtlRun(seed=seed, randomize=randomize, T=60.0)
    env = r.env
    p0 = env.d.xpos[env.id_torso].copy()
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    f0 = R0[:, 0].copy(); f0[2] = 0; f0 /= np.linalg.norm(f0)
    _walk_for(r, 0.8)
    _walk_for(r, 3.6, vx=23.0)
    _walk_for(r, 0.9)
    r.script(wave_short, wave_short.T, 'wave')
    _walk_for(r, 0.4)
    _turn_by(r, math.pi, 23.0)
    _walk_for(r, 0.9)
    _walk_for(r, 3.6, vx=23.0)
    _walk_for(r, 0.8)
    s = dict(r.stats)
    s['states'] = sorted(s['states'])
    s.update(ok=bool(r.upright() and s['falls'] == 0), final_tilt=fs.torso_tilt_deg(env), xy=(float(env.d.qpos[0]), float(env.d.qpos[1])))
    c0 = p0 + f0 * 0.075
    return dict(frames=r.frames, env=env, stats=s, T=r.frames[-1][0], view_override=dict(center=(float(c0[0]), float(c0[1]))))


@hybrid('spin', '转圈圈', 'pirouette  -  one full turn on the spot with the left-turn gait  (x2)', dict(azim=145, elev=-25, dist=0.5, lookat_z=0.05), speed=2.0)
def run_spin(seed=0, randomize=False):
    import fun_ctl
    r = fun_ctl.CtlRun(seed=seed, randomize=randomize, T=40.0)
    env = r.env
    _walk_for(r, 0.8)
    _turn_by(r, 2 * math.pi, 23.0)
    _walk_for(r, 1.0)
    s = dict(r.stats)
    s['states'] = sorted(s['states'])
    s.update(ok=bool(r.upright() and s['falls'] == 0), final_tilt=fs.torso_tilt_deg(env), xy=(float(env.d.qpos[0]), float(env.d.qpos[1])))
    return dict(frames=r.frames, env=env, stats=s, T=r.frames[-1][0])


# ================================================================================================ running
def _check_ok(spec, pl):
    s = pl.stats
    lim = spec['tilt_limit'] if spec['tilt_limit'] is not None else 10.0
    d = pl.env.d
    R = d.xmat[pl.env.id_torso].reshape(3, 3)
    final_tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    gap_ok = spec['family'] != 'emotion' or s['min_gap'] > 0.004          # the torso stays >= 4 mm above the floor while it is pitched
    return bool(s['max_tilt'] <= lim and final_tilt < 6.0 and s['min_z'] > 0.062 and gap_ok), final_tilt


def run_open(name, seed=0, randomize=False, T=None):
    spec = MOTIONS[name]
    T = T or spec['T']
    env = make_env(seed=seed, randomize=randomize, T=T + 1.0)
    pl = Player(env)
    n = int(round(T / DT))
    for i in range(n):
        pl.step(spec['fn'](i * DT), '')
    ok, final_tilt = _check_ok(spec, pl)
    stats = dict(pl.stats)
    stats.update(ok=ok, final_tilt=final_tilt, xy=(float(env.d.qpos[0]), float(env.d.qpos[1])))
    return dict(name=name, title=spec['title'], subtitle=spec['subtitle'], env=env, frames=pl.frames, view=spec['view'], stats=stats, T=T, family=spec['family'])


def run(name, seed=0, randomize=False):
    if name in MOTIONS:
        return run_open(name, seed, randomize)
    h = HYBRIDS[name]
    c = h['runner'](seed=seed, randomize=randomize)
    c.update(name=name, title=h['title'], subtitle=h['subtitle'], view=dict(h['view'], **c.pop('view_override', {})), family='ctl', speed=h['speed'])
    return c


def cmd_list():
    for k, s in MOTIONS.items():
        print('%-10s %-8s %-6s T=%.1fs  %s' % (k, s['title'], s['family'], s['T'], s['subtitle']))
    for k, s in HYBRIDS.items():
        print('%-10s %-8s %-6s %s' % (k, s['title'], s['family'], s['subtitle']))


def cmd_check(names, n):
    names = names or list(MOTIONS) + list(HYBRIDS)
    print('%-10s %6s %8s %12s %10s %9s %6s %8s' % ('motion', 'ok', 'tilt max', 'pitch range', 'hip speed', 'servo sat', 'rms', 'min gap'))
    for nm in names:
        oks, tilts, sp, sat, rms, pmin, pmax, tg, gaps = [], [], [], [], [], [], [], [], []
        for seed in range(n):
            c = run(nm, seed=100 + seed, randomize=True)
            st = c['stats']
            oks.append(st['ok']); tilts.append(st['max_tilt'])
            if nm in MOTIONS:
                sp.append(st['max_hip_speed']); sat.append(st['hip_sat']); rms.append(st['hip_rms']); pmin.append(st['min_pitch']); pmax.append(st['max_pitch']); gaps.append(st['min_gap'])
            elif st.get('t_getup'):
                tg.append(st['t_getup'])
        if nm in MOTIONS:
            print('%-10s %3d/%-3d %7.1f° [%4.0f,%4.0f]° %8.1f/s %8.0f%% %6.2f %6.1fmm' % (nm, sum(oks), n, max(tilts), min(pmin), max(pmax), max(sp), 100 * np.mean(sat), np.mean(rms), min(gaps) * 1000 if min(gaps) < 1 else float('nan')), flush=True)
        else:
            print('%-10s %3d/%-3d %7.1f°   %s' % (nm, sum(oks), n, max(tilts), ('time of standing again: %.1f..%.1f s' % (min(tg), max(tg))) if tg else ''), flush=True)


MAIN = ['wave', 'bow', 'sneeze', 'headbang', 'lookup', 'cheer', 'windmill', 'robotdance', 'swim', 'playdead']
BONUS = ['nod', 'twist', 'fold', 'hello', 'spin', 'medley']


def build_all(n_check=30):
    """everything for the report: the clips (MP4 + GIF + strip), the overview sheet, the randomised check numbers and a small HTML page"""
    import fun_render as fr
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    clips = []
    for k, nm in enumerate(MAIN + BONUS):
        c = run(nm)
        fname = '%02d_%s' % (k + 1, nm) if nm in MAIN else 'bonus_%s' % nm
        c['name'] = fname
        gif = nm != 'medley'
        out = fr.render_clip(c, OUT, gif=gif, strip_times=fr.key_times(c, 8))
        # randomised check
        oks, tilts, sp, sat, rms = [], [], [], [], []
        for seed in range(n_check):
            cc = run(nm, seed=100 + seed, randomize=True)
            st = cc['stats']
            oks.append(bool(st['ok'])); tilts.append(float(st['max_tilt']))
            if 'max_hip_speed' in st:
                sp.append(float(st['max_hip_speed'])); sat.append(float(st['hip_sat'])); rms.append(float(st['hip_rms']))
        st = c['stats']
        summary[nm] = dict(file=fname, title=c['title'], subtitle=c['subtitle'], duration=c['T'], speed=c.get('speed', 1.0), ok_nominal=bool(st['ok']), ok_random='%d/%d' % (sum(oks), n_check),
                           max_tilt_random=max(tilts), hip_speed_max=max(sp) if sp else None, hip_sat_mean=float(np.mean(sat)) if sat else None, hip_rms_mean=float(np.mean(rms)) if rms else None,
                           pitch=[float(st['min_pitch']), float(st['max_pitch'])] if 'min_pitch' in st else None, files={k_: os.path.basename(v) for k_, v in out.items()})
        print(nm, summary[nm]['ok_random'], out, flush=True)
        if nm in MAIN:
            clips.append(c)
    fr.overview(clips, os.path.join(OUT, 'overview.png'))
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    write_index(summary)


EMO_MAIN = EMOTIONS
EMO_EXTRA = ['loving', 'impatient']


def build_emotions(n_check=30):
    """the emotion motions: clips (MP4 + GIF + strip) emo_*.mp4, emotions_overview.png, emotions_summary.json, emotions.html"""
    import fun_render as fr
    os.makedirs(OUT, exist_ok=True)
    summary, clips = {}, []
    for nm in EMO_MAIN + EMO_EXTRA:
        c = run(nm)
        c['name'] = 'emo_' + nm
        c['subtitle'] = '%s' % MOTIONS[nm]['subtitle']
        out = fr.render_clip(c, OUT, strip_times=fr.key_times(c, 8))
        oks, tilts, sp, gaps = [], [], [], []
        for seed in range(n_check):
            st = run(nm, seed=100 + seed, randomize=True)['stats']
            oks.append(bool(st['ok'])); tilts.append(float(st['max_tilt'])); sp.append(float(st['max_hip_speed'])); gaps.append(float(st['min_gap']))
        st = c['stats']
        summary[nm] = dict(file='emo_' + nm, title=c['title'], subtitle=MOTIONS[nm]['subtitle'], face=MOTIONS[nm]['face'], duration=c['T'], ok_random='%d/%d' % (sum(oks), n_check),
                           max_tilt_random=max(tilts), hip_speed_max=max(sp), min_gap_mm=min(gaps) * 1000, pitch=[float(st['min_pitch']), float(st['max_pitch'])],
                           files={k_: os.path.basename(v) for k_, v in out.items()})
        print(nm, summary[nm]['ok_random'], 'gap %.1f mm' % (min(gaps) * 1000), out.get('mp4'), flush=True)
        if nm in EMO_MAIN:
            clips.append(c)
    fr.overview(clips, os.path.join(OUT, 'emotions_overview.png'), frames_per_clip=6)
    json.dump(summary, open(os.path.join(OUT, 'emotions_summary.json'), 'w'), indent=1, ensure_ascii=False)
    rows = ['<h2>情绪动作（撑起站：躯干在髋上前倾后仰，升降控制高低）</h2><div class="grid">']
    for nm in EMO_MAIN + EMO_EXTRA:
        s_ = summary[nm]
        rows.append('<div class="card"><video src="%s" autoplay loop muted playsinline></video><h3>%s <small>%s · 脸：%s</small></h3><p>%s</p><p class="m">随机化仿真 %s 次没倒 · 躯干离地最小 %.1f mm · 髋舵机最高 %.1f rad/s</p></div>' % (
            s_['files']['mp4'], s_['title'], nm, s_['face'], s_['subtitle'], s_['ok_random'], s_['min_gap_mm'], s_['hip_speed_max']))
    rows.append('</div>')
    html = open(os.path.join(OUT, 'index.html')).read() if os.path.exists(os.path.join(OUT, 'index.html')) else ''
    head = html.split('<h2>')[0] if html else '<!doctype html><meta charset="utf-8">'
    open(os.path.join(OUT, 'emotions.html'), 'w').write(head + '\n'.join(rows))


def write_index(summary):
    rows = []
    for group, names in (('十个动作', MAIN), ('加送', BONUS)):
        rows.append('<h2>%s</h2><div class="grid">' % group)
        for nm in names:
            s = summary[nm]
            rows.append('<div class="card"><video src="%s" autoplay loop muted playsinline></video><h3>%s <small>%s</small></h3><p>%s</p><p class="m">随机化仿真 %s 次没倒 · 髋舵机最高 %s rad/s</p></div>' % (
                s['files']['mp4'], s['title'], nm, s['subtitle'], s['ok_random'], ('%.1f' % s['hip_speed_max']) if s['hip_speed_max'] else '-'))
        rows.append('</div>')
    fw = [(f, t, d) for f, t, d in (('fw_bow.mp4', '固件版 鞠躬', 'the firmware supervisor (C code) plays the bow table: STAND -> MOTION -> STAND'),
                                    ('fw_bow_poke.mp4', '固件版 鞠躬 + 被戳', 'poked during the bow: MOTION -> FALLEN -> learned get-up -> STAND')) if os.path.exists(os.path.join(OUT, f))]
    if fw:
        rows.append('<h2>固件里的版本（监督器 C 代码在环）</h2><div class="grid">')
        for f, t, d in fw:
            rows.append('<div class="card"><video src="%s" autoplay loop muted playsinline></video><h3>%s</h3><p>%s</p></div>' % (f, t, d))
        rows.append('</div>')
    html = """<!doctype html><meta charset="utf-8"><title>Tars 好玩的动作 (MuJoCo)</title>
<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#1b1d23;color:#eee;margin:20px}h1{font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px}
.card{background:#262a33;border-radius:10px;padding:10px}video{width:100%%;border-radius:6px}h3{margin:8px 0 2px}small{color:#9aa}p{margin:2px 0;color:#cfd}p.m{color:#9ab;font-size:12px}</style>
<h1>Tars 好玩的动作（MuJoCo 校准仿真）</h1><p>左臂蓝色、右臂橙色（真机是黑色，这里上色方便看）。 概览图：<a style="color:#8cf" href="overview.png">overview.png</a></p>%s""" % '\n'.join(rows)
    open(os.path.join(OUT, 'index.html'), 'w').write(html)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'check', 'render', 'strips', 'all', 'index', 'emotions'])
    ap.add_argument('names', nargs='*')
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    if a.cmd == 'list':
        cmd_list()
    elif a.cmd == 'check':
        cmd_check(a.names, a.n)
    elif a.cmd == 'strips':
        import fun_render as fr
        for nm in (a.names or list(MOTIONS) + list(HYBRIDS)):
            c = run(nm)
            os.makedirs('/tmp/fun', exist_ok=True)
            fr.render_strip(c, '/tmp/fun/s_%s.png' % nm, fr.key_times(c, 8), 4, (300, 225))
            print(nm, 'ok' if c['stats']['ok'] else 'NOT OK', flush=True)
    elif a.cmd == 'all':
        build_all(a.n)
    elif a.cmd == 'emotions':
        build_emotions(a.n)
    elif a.cmd == 'index':                                   # only rewrite index.html (e.g. after rendering the fw_* clips)
        write_index(json.load(open(os.path.join(OUT, 'summary.json'))))
    else:
        import fun_render as fr
        for nm in (a.names or list(MOTIONS) + list(HYBRIDS)):
            c = run(nm)
            out = fr.render_clip(c, OUT, strip_times=fr.key_times(c, 8))
            print(nm, out, flush=True)
