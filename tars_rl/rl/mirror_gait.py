"""Start point for a BACKWARD gait search from a forward gait (mirror trick): the front/back mirror image of a forward gait is a backward gait.
Hip pattern (q0, a1, b1, a2, b2) and the lift feedback (kp_z, kd_z) change sign (pitch and pitch rate flip in the mirror world, the hip feedback
kp_h/kd_h and the lift pattern keep their sign); frequency and everything else stay.  The robot is not exactly symmetric, so the result is only a start point.

usage: python rl/mirror_gait.py runs/walk_cma_cal/fwd_s1.json runs/walk_cma_cal/back_start_mirror_s1.npy [--check runs/walk_cma_cal/back_start_mirror_c4.npy]
  writes the normalised parameter vector in the DEFAULT ranges of rl/walk_cma.py (what `walk_cma_cal.py --x0` expects; it applies --fmax/--fmin itself)"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
sys.path.insert(0, os.path.join(ROOT, 'rl'))
import walk_cma as W  # noqa: E402

FLIP = ('q0', 'a1', 'b1', 'a2', 'b2', 'kp_z', 'kd_z')


def mirror(p):
    return {k: (-v if k in FLIP else v) for k, v in p.items()}


def encode(p):
    v = np.array([p[k] for k in W.NAMES], float)
    return np.clip((v - W.LO) / (W.HI - W.LO) * 2 - 1, -1, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('fwd_json')
    ap.add_argument('out_npy')
    ap.add_argument('--check', default='', help='an existing start vector to compare with (sanity check of the sign convention)')
    a = ap.parse_args()
    p = json.load(open(a.fwd_json))['params']
    x = encode(mirror(p))
    np.save(a.out_npy, x)
    print('wrote', a.out_npy, {k: round(float(v), 4) for k, v in zip(W.NAMES, W.decode(x))})
    if a.check:
        y = np.load(a.check)
        print('max |difference| to', a.check, '=', float(np.abs(x - y).max()))
