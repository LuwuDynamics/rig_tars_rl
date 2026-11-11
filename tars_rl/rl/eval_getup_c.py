"""The firmware get-up logic (C: tars_ctrl.c with the exported network) in the randomised simulator, for the three get-up modes:
  kf  : keyframes only (what runs on the robot today)           alt : network on the 1st and 3rd attempt, keyframes on the 2nd           nn : network only
The supervisor sees exactly what the firmware sees (its own Mahony filter on the simulated accelerometer / gyro, saturating like the QMI8658) and drives
the simulated servos at 50 Hz; fall detection, the wait for stillness, retries and the hand-over to STAND are all the firmware's own.
Success = the simulator's own test (upright, still, arms nominal for 1 s).

usage: python rl/eval_getup_c.py [--n 100] [--T 30] [--modes kf,alt,nn] [--plants cal,orig] [--stress none]
"""
import argparse
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('sim', 'rl', 'calib'):
    sys.path.insert(0, os.path.join(ROOT, p))
from tars_env import DT                                    # noqa: E402
from getup_env import GetupEnv, HIP, LIFT                  # noqa: E402
import eval_cal                                            # noqa: E402
import c_in_loop                                           # noqa: E402
from test_c_loop import make_ctl                           # noqa: E402
from eval_getup import STRESS                              # noqa: E402

KEYFRAMES = os.path.join(ROOT, 'data/getup_cma/back_tau1.0_rate8.0.npy')
MODES = {'kf': 0, 'alt': 1, 'nn': 2}
_CTL = {}


def _init_worker():
    c_in_loop.build = lambda: None


def job(args):
    mode, plant, kind, seed, T, stress = args
    init = None if kind == 'random' else dict(kind=kind, hip=(0.0, 0.0))
    if 'k' not in _CTL:
        _CTL['k'] = make_ctl(None, KEYFRAMES)
    ctl = _CTL['k']
    ctl.set_getup_mode(MODES[mode])
    cal = eval_cal.calibrated_kwargs(lag=1, lag_jitter=1)
    cal.pop('ctrl_dt', None)
    cal.update(STRESS[stress])
    env = GetupEnv(seed=seed, plant_cal=cal, plant_mix=(1.0 if plant == 'cal' else 0.0), continue_after_success=False, max_steps=int(round(T / DT)), action_repeat=1,
                   rate_hip=1e3, rate_lift=1e3, imu_clip=True, com_shift=cal.get('com_shift'), getup_init=init)
    ctl.reset(2.0)
    fifo = deque(maxlen=64)

    def hook(acc, gyr):
        fifo.append((np.array(acc), np.array(gyr)))
        lag = env.pr.get('imu_lag', 0) if hasattr(env, 'pr') else 0
        a_, g_ = fifo[-1 - min(lag, len(fifo) - 1)]
        ctl.imu(a_, g_, 0.01)
    env.imu_hook = hook
    env.reset()
    ctl.set_kp(env.filter.kp)
    lag = env.pr.get('imu_lag', 0)
    states = []
    for i in range(env.max_steps):
        ql, qr, z, st = ctl.step(fifo[-1 - min(lag, len(fifo) - 1)][1], DT, 0.0, 0.0)
        obs, r, d, info = env.step(np.array([ql / HIP, qr / HIP, z / LIFT]))
        states.append(st)
        if d:
            break
    nn_att = sum(1 for k in range(1, len(states)) if states[k] == 11 and states[k - 1] != 11)
    kf_att = sum(1 for k in range(1, len(states)) if states[k] == 6 and states[k - 1] != 6)
    return env.t_success if env.succeeded else None, nn_att, kf_att, states[-1] == 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--T', type=float, default=30.0)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--modes', default='kf,alt,nn')
    ap.add_argument('--plants', default='cal,orig')
    ap.add_argument('--kinds', default='back,front,random')
    ap.add_argument('--stress', default='none')
    ap.add_argument('--seed0', type=int, default=6000)
    a = ap.parse_args()
    c_in_loop.build()
    pool = Pool(a.procs, initializer=_init_worker)
    print('### firmware get-up logic (C) in simulation, stress=%s: N=%d, horizon %.0f s\n' % (a.stress, a.n, a.T))
    print('| plant | start | mode | success <=12 s | success <=%.0f s | time to stand (mean / median) | network attempts | keyframe attempts | FAILED |' % a.T)
    print('|---|---|---|---|---|---|---|---|---|')
    for plant in a.plants.split(','):
        for kind in a.kinds.split(','):
            for mode in a.modes.split(','):
                res = pool.map(job, [(mode, plant, kind, a.seed0 + k, a.T, a.stress if plant == 'cal' else 'none') for k in range(a.n)])
                ok = [r[0] for r in res if r[0] is not None]
                n12 = sum(1 for t in ok if t <= 12.0)
                print('| %s | %s | %s | %d/%d | %d/%d | %s | %.2f | %.2f | %d |' % (plant, kind, mode, n12, a.n, len(ok), a.n,
                      ('%.1f s / %.1f s' % (np.mean(ok), np.median(ok))) if ok else '-', np.mean([r[1] for r in res]), np.mean([r[2] for r in res]), sum(1 for r in res if r[3])), flush=True)
    pool.close()


if __name__ == '__main__':
    main()
