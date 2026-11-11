"""Is a get-up physically possible with 3 SCS0009 servos?  Open-loop trajectory optimisation (CMA-ES) in the
simulator, no learning.  A YES here means RL / scripted get-up is worth building; a NO is a hardware limit.

usage: python rl/getup_cma.py back [--tau 1.0] [--gens 80] [--pop 32]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import cma
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
from tars_env import TarsEnv, DT                                  # noqa: E402

K = 14            # keyframes
T_END = 7.0
HIP = 2.4         # rad
LIFT = 0.014


def traj(x, t, tend=None):
    """x: [hip_k (K), lift_k (K)] normalised in [-1,1] -> (hip, lift) at time t (linear interpolation)"""
    tk = np.linspace(0, tend or T_END, K)
    h = np.interp(t, tk, np.clip(x[:K], -1, 1))
    z = np.interp(t, tk, np.clip(x[K:2 * K], -1, 1))
    return h, z


def rollout(args):
    x, kind, tau_scale, seed, randomize, hips0, rate, tend, strict = args
    env = TarsEnv(seed=seed, task='getup', hip_scale=HIP, lift_scale=LIFT, max_steps=int(tend / DT) + 1,
                  randomize=randomize, difficulty=1.0, push_prob_per_s=0.0, getup_strict=strict, getup_init=dict(kind=kind, hip=hips0))
    if tau_scale != 1.0 and not randomize:
        env.m.actuator_forcerange[:] = env._nom['forcerange'] * tau_scale
    env.reset()
    ret, succ, tsucc = 0.0, False, None
    best_up = -1.0
    hc, zc = 0.0, 0.0                                   # slew-rate-limited command (= servo speed register)
    for i in range(env.max_steps):
        h, z = traj(x, i * DT, tend)
        hc += float(np.clip(h * HIP - hc, -rate * DT, rate * DT))
        zc += float(np.clip(z * LIFT - zc, -0.25 * DT, 0.25 * DT))
        o, r, d, info = env.step(np.array([hc / HIP, hc / HIP, zc / LIFT]))
        ret += r
        best_up = max(best_up, env.d.xmat[env.id_torso].reshape(3, 3)[2, 2])
        if d:
            succ, tsucc = True, i * DT
            break
    score = (800.0 + (tend - tsucc) * 20.0) if succ else ret
    return score, succ, tsucc, best_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', default='back')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--gens', type=int, default=80)
    ap.add_argument('--pop', type=int, default=32)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--ndr', type=int, default=3, help='randomised rollouts per candidate in the objective')
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--rate', type=float, default=4.0, help='max hip speed rad/s (servo speed register)')
    ap.add_argument('--tend', type=float, default=7.0)
    ap.add_argument('--lenient', action='store_true', help='success = upright & stable on the torso feet (arms may stay raised)')
    ap.add_argument('--out', default='runs/getup_cma')
    a = ap.parse_args()
    global T_END
    T_END = a.tend
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    rng = np.random.default_rng(0)
    x0 = np.zeros(2 * K)
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {'popsize': a.pop, 'seed': 1, 'verbose': -9, 'bounds': [-1, 1]})
    pool = Pool(a.procs)
    best = (-1e9, None)
    t0 = time.time()
    for g in range(a.gens):
        xs = es.ask()
        jobs = []
        for x in xs:
            jobs.append((x, a.kind, a.tau, 0, False, (0.0, 0.0), a.rate, a.tend, not a.lenient))
            for k in range(a.ndr):
                jobs.append((x, a.kind, a.tau, 1000 * g + k, True, (0.0, 0.0), a.rate, a.tend, not a.lenient))
        flat = pool.map(rollout, jobs)
        m_ = 1 + a.ndr
        res = [(float(np.mean([flat[i * m_ + j][0] for j in range(m_)])), all(flat[i * m_ + j][1] for j in range(m_)),
                flat[i * m_][2], max(flat[i * m_ + j][3] for j in range(m_))) for i in range(len(xs))]
        rets = np.array([r[0] for r in res])
        es.tell(xs, (-rets).tolist())
        i = int(np.argmax(rets))
        if rets[i] > best[0]:
            best = (rets[i], xs[i].copy(), res[i])
        nsucc = sum(r[1] for r in res)   # candidates that succeeded in ALL conditions
        if g % 5 == 0 or g == a.gens - 1:
            print(f'gen {g:3d}  best ret {rets.max():7.1f} (all-time {best[0]:7.1f})  succ {nsucc}/{len(res)}  best_up {max(r[3] for r in res):.2f}  {time.time()-t0:.0f}s', flush=True)
    xb = best[1]
    print('best:', best[2])
    np.save(os.path.join(ROOT, a.out, f'{a.kind}_tau{a.tau}_rate{a.rate}.npy'), xb)
    # robustness of the best open-loop motion under randomisation
    n = 40
    res = pool.map(rollout, [(xb, a.kind, 1.0, 100 + k, True, (0.0, 0.0), a.rate, a.tend, not a.lenient) for k in range(n)])
    print(f'robustness under domain randomisation: success {sum(r[1] for r in res)}/{n}')
    pool.close()


if __name__ == '__main__':
    main()
