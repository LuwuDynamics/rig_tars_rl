"""Refine the open-loop get-up keyframes in the REAL-CALIBRATED simulator (servo fit + command delay + IMU lag from the 2026-10-28 real data).
Starts from the keyframes that are flashed on the robot (data/getup_cma/back_tau1.0_rate8.0.npy); the originals are untouched.
usage: python rl/getup_cma_cal.py back --gens 60 --pop 32 --procs 9 --tag _c1 --sigma 0.12"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import cma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
sys.path.insert(0, os.path.join(ROOT, 'calib'))
from tars_env import TarsEnv, DT                      # noqa: E402
import getup_cma as G                                   # noqa: E402
from eval_cal import calibrated_kwargs                  # noqa: E402

ENV_KW = {}


def rollout(args):
    x, kind, seed, randomize, rate, tend = args
    kw = dict(ENV_KW)
    env = TarsEnv(seed=seed, task='getup', hip_scale=G.HIP, lift_scale=G.LIFT, max_steps=int(tend / DT) + 1, randomize=randomize, difficulty=1.0, push_prob_per_s=0.0,
                 getup_strict=True, getup_init=dict(kind=kind, hip=(0.0, 0.0)), tau_nominal=1.3, **kw)
    env.reset()
    ret, succ, tsucc, best_up = 0.0, False, None, -1.0
    hc, zc = 0.0, 0.0
    for i in range(env.max_steps):
        h, z = G.traj(x, i * DT, tend)
        hc += float(np.clip(h * G.HIP - hc, -rate * DT, rate * DT))
        zc += float(np.clip(z * G.LIFT - zc, -0.25 * DT, 0.25 * DT))
        o, r, d, info = env.step(np.array([hc / G.HIP, hc / G.HIP, zc / G.LIFT]))
        ret += r
        best_up = max(best_up, env.d.xmat[env.id_torso].reshape(3, 3)[2, 2])
        if d:
            succ, tsucc = True, i * DT
            break
    score = (800.0 + (tend - tsucc) * 20.0) if succ else ret
    return score, succ, tsucc, best_up


def init_worker(kw):
    ENV_KW.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', default='back')
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--pop', type=int, default=32)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--ndr', type=int, default=6)
    ap.add_argument('--sigma', type=float, default=0.12)
    ap.add_argument('--rate', type=float, default=8.0)
    ap.add_argument('--tend', type=float, default=7.0)
    ap.add_argument('--x0', default='data/getup_cma/back_tau1.0_rate8.0.npy')
    ap.add_argument('--tag', default='_c1')
    ap.add_argument('--out', default='runs/getup_cma_cal')
    a = ap.parse_args()
    envkw = calibrated_kwargs()
    ENV_KW.update(envkw)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    x0 = np.load(os.path.join(ROOT, a.x0))
    pool = Pool(a.procs, initializer=init_worker, initargs=(envkw,))
    ref = pool.map(rollout, [(x0, a.kind, 900 + k, True, a.rate, a.tend) for k in range(60)])
    print(f'START POINT (flashed keyframes) in the calibrated simulator: success {sum(r[1] for r in ref)}/60 within {a.tend:.0f} s (single attempt, open loop)', flush=True)
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {'popsize': a.pop, 'seed': 2, 'verbose': -9, 'bounds': [-1, 1]})
    best = (-1e9, None)
    t0 = time.time()
    for g in range(a.gens):
        xs = es.ask()
        jobs = []
        for x in xs:
            for k in range(a.ndr):
                jobs.append((x, a.kind, 1000 * g + k, True, a.rate, a.tend))
        flat = pool.map(rollout, jobs)
        rets = np.array([np.mean([flat[i * a.ndr + j][0] for j in range(a.ndr)]) for i in range(len(xs))])
        succ = np.array([np.mean([flat[i * a.ndr + j][1] for j in range(a.ndr)]) for i in range(len(xs))])
        es.tell(xs, (-rets).tolist())
        i = int(np.argmax(rets))
        if rets[i] > best[0]:
            best = (rets[i], xs[i].copy(), succ[i])
            np.save(os.path.join(ROOT, a.out, f'{a.kind}{a.tag}_ckpt.npy'), best[1])
        if g % 5 == 0 or g == a.gens - 1:
            print(f'gen {g:3d} best ret {rets.max():7.1f} (all-time {best[0]:7.1f}, success {best[2]:.2f})  mean success {succ.mean():.2f}  {time.time()-t0:.0f}s', flush=True)
    xb = best[1]
    np.save(os.path.join(ROOT, a.out, f'{a.kind}{a.tag}.npy'), xb)
    res = pool.map(rollout, [(xb, a.kind, 5000 + k, True, a.rate, a.tend) for k in range(100)])
    res0 = pool.map(rollout, [(x0, a.kind, 5000 + k, True, a.rate, a.tend) for k in range(100)])
    print(f'VALIDATION (100 fresh randomised calibrated seeds, single open-loop attempt of {a.tend:.0f} s): new keyframes {sum(r[1] for r in res)}/100, flashed keyframes {sum(r[1] for r in res0)}/100')
    pool.close()


if __name__ == '__main__':
    main()
