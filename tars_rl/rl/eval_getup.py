"""Compare get-up controllers in the same randomised simulation (same seeds => same plant randomisation and start pose):
  keyframes : the firmware supervisor (C code tars_ctrl.c, CMA keyframes + two-stage finish + up to 3 retries), exactly what runs on the robot
  <ckpt>    : an actor network (.pt from rl/ppo_getup.py or rl/bc_getup.py) acting every 0.1 s in GetupEnv
Start poses: back / front (arms at the nominal pose, yaw 0: the numbers of calib/eval_cal.py) and 'random' (the training distribution: back / front / leaning, random arms and yaw).
Plants: orig (original simulator) and cal (real-calibrated: fitted servo, 15.7 ms command delay, IMU lag).  Success = upright, still and arms nominal for 1 s.

usage: python rl/eval_getup.py --policy runs/getup_ppo_a/best.pt [--policy other.pt ...] [--n 60] [--T 30] [--out runs/eval_getup_a.md] [--no-keyframes]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('sim', 'rl', 'calib'):
    sys.path.insert(0, os.path.join(ROOT, p))
from tars_env import DT                                    # noqa: E402
from getup_env import GetupEnv                             # noqa: E402
import eval_cal                                            # noqa: E402
import c_in_loop                                           # noqa: E402
from test_c_loop import make_ctl                           # noqa: E402
from evaluate import load_policy                           # noqa: E402

KEYFRAMES = os.path.join(ROOT, 'data/getup_cma/back_tau1.0_rate8.0.npy')
_CTL = {}
_POL = {}


def _init_worker():
    c_in_loop.build = lambda: None                         # the parent compiled the C controller once


STRESS = {                                                 # conditions NOT seen in training (applied on top of the calibrated plant)
    'none': {},
    'weak': dict(tau_range=(0.80, 0.95)),                                    # servo torque 20-35 % lower (low battery)
    'slow': dict(cmd_delay_ms=32.0, cmd_delay_jitter_ms=6.0, imu_lag_samples=3, imu_lag_jitter=1),    # twice the command delay, 3 IMU samples old
    'com': dict(com_shift=(0.010, 0.0, 0.004)),                              # centre of mass 10 mm forward, 4 mm up
    'comb': dict(com_shift=(-0.010, 0.0, 0.004)),                            # centre of mass 10 mm backward
}


def cal_kw(stress='none'):
    kw = eval_cal.calibrated_kwargs(lag=1, lag_jitter=1)
    kw.pop('ctrl_dt', None)
    kw.update(STRESS[stress])
    return kw


def job(args):
    who, plant, kind, seed, T, stress = args
    init = None if kind == 'random' else dict(kind=kind, hip=(0.0, 0.0))
    if who == 'keyframes':
        if 'k' not in _CTL:
            _CTL['k'] = make_ctl(None, KEYFRAMES)
        r = eval_cal.run_episode(_CTL['k'], seed, T, 'getup', (0, 0), 0.0, init, cal_kw(stress) if plant == 'cal' else None)
        return r['succ_t']
    if who not in _POL:
        _POL[who] = load_policy(who)
    fn, ck = _POL[who]
    a = ck['args']
    pk = cal_kw(stress)
    kw = dict(seed=seed, plant_cal=pk, plant_mix=(1.0 if plant == 'cal' else 0.0), max_steps=int(round(T / DT)), com_shift=pk.get('com_shift'),
              getup_init=init, action_repeat=a.get('repeat', 5), time_obs=bool(a.get('time_obs', 1)), rate_hip=a.get('rate_hip', 8.0), rate_lift=a.get('rate_lift', 0.25))
    kw.update({k: v for k, v in json.loads(a.get('env_kw') or '{}').items() if k in ('obs_q_mode', 'imu_clip')})      # observation options of the training
    kw['continue_after_success'] = False
    env = GetupEnv(**kw)
    obs, _ = env.reset()
    while True:
        obs, r, d, info = env.step(fn(obs))
        if d or info['truncated']:
            break
    return env.t_success if env.succeeded else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', action='append', default=[])
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--T', type=float, default=30.0)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--seed0', type=int, default=6000)
    ap.add_argument('--no-keyframes', action='store_true')
    ap.add_argument('--plants', default='cal,orig')
    ap.add_argument('--kinds', default='back,front,random')
    ap.add_argument('--stress', default='none', help='none | weak | slow | com | comb (calibrated plant only): conditions beyond the training range')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    c_in_loop.build()
    pool = Pool(a.procs, initializer=_init_worker)
    who_list = ([] if a.no_keyframes else ['keyframes']) + a.policy
    lines = ['### get-up in simulation (stress: %s): N=%d per cell, horizon %.0f s, success = upright + still + arms nominal for 1 s\n' % (a.stress, a.n, a.T),
             '| plant | start | controller | success <=12 s | success <=%.0f s | time to stand (mean / median of successes) |' % a.T,
             '|---|---|---|---|---|---|']
    for plant in a.plants.split(','):
        for kind in a.kinds.split(','):
            for who in who_list:
                t0 = time.time()
                res = pool.map(job, [(who, plant, kind, a.seed0 + k, a.T, a.stress) for k in range(a.n)])
                ok = [r for r in res if r is not None]
                n12 = sum(1 for r in ok if r <= 12.0)
                name = 'keyframes (firmware)' if who == 'keyframes' else os.path.join(os.path.basename(os.path.dirname(who)), os.path.basename(who))
                line = '| %s | %s | %s | %d/%d (%.0f%%) | %d/%d (%.0f%%) | %s |' % (
                    plant, kind, name, n12, a.n, 100.0 * n12 / a.n, len(ok), a.n, 100.0 * len(ok) / a.n,
                    ('%.1f s / %.1f s' % (np.mean(ok), np.median(ok))) if ok else '-')
                lines.append(line)
                print(line, '  (%.0fs)' % (time.time() - t0), flush=True)
    pool.close()
    if a.out:
        open(os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
