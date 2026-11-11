"""Behaviour cloning of the firmware get-up (C supervisor: CMA keyframes + retries) into an actor network that has only what the ESP32 can measure
(attitude, gyro, servo targets, previous action, elapsed time), acting every 0.1 s.  Used as the starting point of PPO fine-tuning (rl/ppo_getup.py --init).

usage:  python rl/bc_getup.py collect --n 800 --out runs/getup_bc/demos.npz        (demonstrations from the C controller in the randomised simulator)
        python rl/bc_getup.py train --demos runs/getup_bc/demos.npz --out runs/getup_bc/bc.pt
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
from tars_env import DT, OBS_DIM, PRIV_DIM, ACT_DIM        # noqa: E402
from getup_env import GetupEnv, HIP, LIFT                  # noqa: E402
import eval_cal                                            # noqa: E402
import c_in_loop                                           # noqa: E402
from test_c_loop import make_ctl                           # noqa: E402

KEYFRAMES = os.path.join(ROOT, 'data/getup_cma/back_tau1.0_rate8.0.npy')
K = 5                                                      # base steps per decision (0.1 s)
_CTL = {}


def _init_worker():
    c_in_loop.build = lambda: None


def episode(args):
    seed, plant, T = args
    if 'k' not in _CTL:
        _CTL['k'] = make_ctl(None, KEYFRAMES)
    ctl = _CTL['k']
    cal = eval_cal.calibrated_kwargs(lag=1, lag_jitter=1)
    cal.pop('ctrl_dt', None)
    env = GetupEnv(seed=seed, plant_cal=cal, plant_mix=(1.0 if plant == 'cal' else 0.0), continue_after_success=True, max_steps=int(round(T / DT)),
                   action_repeat=1, rate_hip=1e3, rate_lift=1e3, getup_init=None)       # the supervisor does its own rate limiting: no extra limit, no repeat
    ctl.reset(2.0)
    fifo = deque(maxlen=64)

    def hook(acc, gyr):
        fifo.append((np.array(acc), np.array(gyr)))
        lag = env.pr.get('imu_lag', 0) if hasattr(env, 'pr') else 0
        a_, g_ = fifo[-1 - min(lag, len(fifo) - 1)]
        ctl.imu(a_, g_, 0.01)
    env.imu_hook = hook
    obs, _ = env.reset()
    ctl.set_kp(env.filter.kp)
    lag = env.pr.get('imu_lag', 0)
    O, A, S = [obs.copy()], [], []                         # O[i] = observation before base step i, A[i] = action applied at base step i, S[i] = supervisor state after it
    ok = False
    for i in range(env.max_steps):
        g_ = fifo[-1 - min(lag, len(fifo) - 1)][1]
        ql, qr, z, st = ctl.step(g_, DT, 0.0, 0.0)
        a = np.array([ql / HIP, qr / HIP, z / LIFT])
        obs, r, d, info = env.step(a)
        A.append(a)
        S.append(st)
        O.append(obs.copy())
        if env.succeeded and env.t > env.t_success + 2.0:        # keep 2 s of the standing phase: the policy has to hold still there
            ok = True
            break
    if not ok:
        return None
    n = len(A)
    # the time axis starts when the supervisor starts the get-up (state GETUP1 = 6): the fall detection / waiting-for-still part before it is not part of the policy's job.
    # Only first-attempt successes are used (a retry would restart the supervisor's clock, which the policy cannot know).
    ent = [i for i in range(n) if S[i] == 6 and (i == 0 or S[i - 1] != 6)]
    if len(ent) != 1:
        return None
    i0 = max(0, ent[0] - 1)
    # decisions every K base steps: observation at the start of the interval, target = the supervisor's command at the END of the interval
    # (the servo is rate limited in the same way, so holding that target for 0.1 s reproduces the supervisor's motion)
    idx = np.arange(i0, n - K + 1, K)
    rows = []
    for i in idx:
        o = O[i].copy()
        o[0] = min((i - i0) * DT / 10.0, 1.5)
        rows.append(o)
    return np.array(rows, np.float32), np.array([np.clip(A[i + K - 1], -1, 1) for i in idx], np.float32), env.pose_kind, plant


def collect(a):
    c_in_loop.build()
    pool = Pool(a.procs, initializer=_init_worker)
    jobs = [(a.seed0 + k, 'cal' if k % 2 == 0 else 'orig', a.T) for k in range(a.n)]
    t0 = time.time()
    res = pool.map(episode, jobs, chunksize=4)
    pool.close()
    good = [r for r in res if r is not None]
    print('episodes: %d, successful: %d (%.0f%%), %.0f s' % (len(res), len(good), 100.0 * len(good) / len(res), time.time() - t0))
    O = np.concatenate([g[0] for g in good])
    A = np.concatenate([g[1] for g in good])
    kinds = np.concatenate([[g[2]] * len(g[0]) for g in good])
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, a.out), obs=O, act=A, kind=kinds)
    print('samples:', len(O), ' obs', O.shape, ' act', A.shape)


def train(a):
    import torch
    import torch.nn as nn
    from ppo import ActorCritic, RunningNorm
    torch.manual_seed(0)
    d = np.load(os.path.join(ROOT, a.demos))
    O, A = d['obs'], d['act']
    n = len(O)
    perm = np.random.default_rng(0).permutation(n)
    nv = n // 10
    va, tr = perm[:nv], perm[nv:]
    onorm = RunningNorm(OBS_DIM)
    onorm.update(O.astype(np.float64))
    on = lambda x: np.clip((x - onorm.mean) / np.sqrt(onorm.var + 1e-8), -10, 10).astype(np.float32)
    Xt, Yt = torch.from_numpy(on(O[tr])), torch.from_numpy(A[tr])
    Xv, Yv = torch.from_numpy(on(O[va])), torch.from_numpy(A[va])
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM, (128, 128), (256, 256), a.log_std0)
    opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_sd = 1e9, None
    for ep in range(a.epochs):
        p = torch.randperm(len(Xt))
        tl = 0.0
        for mb in p.chunk(max(1, len(Xt) // 512)):
            loss = ((net.actor(Xt[mb]) - Yt[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(mb)
        sched.step()
        with torch.no_grad():
            vl = ((net.actor(Xv) - Yv) ** 2).mean().item()
        if vl < best:
            best, best_sd = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 20 == 0 or ep == a.epochs - 1:
            print('epoch %3d  train mse %.5f  val mse %.5f' % (ep, tl / len(Xt), vl), flush=True)
    net.load_state_dict(best_sd)
    ck = dict(net=net.state_dict(), onorm=dict(mean=onorm.mean, var=onorm.var, count=onorm.count), args=dict(
        actor_hidden=[128, 128], critic_hidden=[256, 256], log_std0=a.log_std0, task='getup', repeat=K, time_obs=1, rate_hip=8.0, rate_lift=0.25, env_kw=''), it=0)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    torch.save(ck, os.path.join(ROOT, a.out))
    print('saved', a.out, ' best val mse %.5f' % best)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('collect')
    c.add_argument('--n', type=int, default=800)
    c.add_argument('--T', type=float, default=30.0)
    c.add_argument('--procs', type=int, default=3)
    c.add_argument('--seed0', type=int, default=20000)
    c.add_argument('--out', default='runs/getup_bc/demos.npz')
    t = sp.add_parser('train')
    t.add_argument('--demos', default='runs/getup_bc/demos.npz')
    t.add_argument('--out', default='runs/getup_bc/bc.pt')
    t.add_argument('--epochs', type=int, default=200)
    t.add_argument('--log_std0', type=float, default=-1.2)
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else train(a)
