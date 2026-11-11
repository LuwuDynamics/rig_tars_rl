"""Evaluate a trained policy (and the firmware's open-loop gait as baseline) in the randomised simulator.

usage:
  python rl/evaluate.py walk runs/walk_a/best.pt            # command tracking + push sweep + baseline comparison
  python rl/evaluate.py getup runs/getup_a/best.pt          # success rate per fallen pose
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
sys.path.insert(0, os.path.join(ROOT, 'rl'))
from tars_env import TarsEnv, DT                                  # noqa: E402
from baseline_gait import gait_targets                            # noqa: E402
from ppo import ActorCritic, RunningNorm                          # noqa: E402
from tars_env import OBS_DIM, PRIV_DIM, ACT_DIM                   # noqa: E402


def load_policy(path):
    ck = torch.load(path, weights_only=False)
    a = ck['args']
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM, tuple(a['actor_hidden']), tuple(a['critic_hidden']), a['log_std0'])
    net.load_state_dict(ck['net'])
    net.eval()
    mean, var = ck['onorm']['mean'], ck['onorm']['var']

    def fn(obs):
        x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        with torch.no_grad():
            return net.actor(torch.from_numpy(x)).numpy()
    return fn, ck


def baseline_fn(env, amp_counts, turn_counts=0.0):
    """firmware open-loop gait expressed as normalised env actions (uses wall-clock = env.t)."""
    def fn(obs):
        q1, q2, z = gait_targets(env.t, amp_counts, turn_counts)
        return np.array([q2 / env.hip_scale, q1 / env.hip_scale, z / env.lift_scale])
    return fn


def run(task_kw, policy_ctor, n_eps, seed0, T_s, cmd=None, push=None, push_t=2.0, randomize=True):
    """policy_ctor(env)->callable(obs)->action.  returns dict of metrics."""
    falls, speeds, wzs, tilts = 0, [], [], []
    for k in range(n_eps):
        kw = dict(task_kw)
        kw.update(max_steps=int(T_s / DT), randomize=randomize, difficulty=1.0)
        if cmd is not None:
            kw['cmd_fixed'] = cmd
        if push is not None:
            ang, imp = push
            kw.update(push_fixed=(ang, imp), push_prob_per_s=1e-3)
        else:
            kw.update(push_prob_per_s=0.0)
        env = TarsEnv(seed=seed0 + k, **kw)
        obs, _ = env.reset()
        if push is not None:
            env.next_push = push_t
        pol = policy_ctor(env)
        fell = False
        xs = []
        for i in range(env.max_steps):
            a = pol(obs)
            obs, r, d, info = env.step(a)
            if i * DT >= 2.0 and 'vf' in info:
                xs.append((info['vf'], info['wz']))
            if d:
                fell = True
                break
        falls += fell
        tilts.append(env.max_tilt_seen)
        if xs:
            xs = np.array(xs)
            speeds.append(xs[:, 0].mean())
            wzs.append(xs[:, 1].mean())
    return dict(fall=falls / n_eps, vf=float(np.mean(speeds)) if speeds else float('nan'),
                wz=float(np.mean(wzs)) if wzs else float('nan'), tilt=float(np.mean(tilts)))


def eval_walk(path, n):
    fn, ck = load_policy(path)
    pol = lambda env: fn
    print('=== command tracking (randomised sim, no pushes, 10 s, deterministic policy) ===')
    print(f'{"cmd vx":>8} {"cmd wz":>7} | {"RL vf":>7} {"RL wz":>7} {"fall":>5} || firmware gait (amp counts -> vf wz fall)')
    for vx, wz, amp, turn in [(0.0, 0.0, 0.0, 0.0), (0.02, 0.0, 15, 0), (0.04, 0.0, 25, 0), (0.06, 0.0, 35, 0),
                              (-0.03, 0.0, -20, 0), (0.0, 0.5, 0, 15), (0.0, -0.5, 0, -15), (0.03, 0.4, 0, 0)]:
        r = run({}, pol, n, 100, 10.0, cmd=(vx, wz))
        b = run({}, lambda env: baseline_fn(env, amp, turn), n, 100, 10.0, cmd=(0, 0)) if (amp or turn) else \
            run({}, lambda env: (lambda o: np.zeros(3)), n, 100, 10.0, cmd=(0, 0))
        print(f'{vx:8.3f} {wz:7.2f} | {r["vf"]:7.3f} {r["wz"]:7.2f} {r["fall"]:5.2f} || amp {amp:4.0f}/{turn:3.0f} -> {b["vf"]:7.3f} {b["wz"]:6.2f} {b["fall"]:5.2f}')
    print()
    print('=== push recovery: single horizontal impulse at t=2 s on the torso, then 6 s observation ===')
    print(f'{"impulse Ns":>10} {"dir":>5} | RL fall(stand) RL fall(walk 0.04) || baseline fall(stand) baseline fall(walk amp25)')
    for imp in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        for ang, name in [(0.0, 'fwd'), (math.pi, 'back')]:
            r0 = run({}, pol, n, 200, 8.0, cmd=(0, 0), push=(ang, imp))
            r1 = run({}, pol, n, 200, 8.0, cmd=(0.04, 0), push=(ang, imp))
            b0 = run({}, lambda env: (lambda o: np.zeros(3)), n, 200, 8.0, cmd=(0, 0), push=(ang, imp))
            b1 = run({}, lambda env: baseline_fn(env, 25, 0), n, 200, 8.0, cmd=(0, 0), push=(ang, imp))
            print(f'{imp:10.2f} {name:>5} | {r0["fall"]:13.2f} {r1["fall"]:16.2f} || {b0["fall"]:19.2f} {b1["fall"]:22.2f}')


def eval_getup(path, n):
    fn, ck = load_policy(path)
    res = {}
    for kind in ['back', 'front', 'lean_f', 'lean_b']:
        ok, times = 0, []
        for k in range(n):
            env = TarsEnv(seed=500 + k, task='getup', max_steps=500, randomize=True, difficulty=1.0, push_prob_per_s=0.0)
            # force the pose kind by resampling until it matches
            for _ in range(50):
                env._init_state()
                if env.pose_kind == kind:
                    break
            obs = env._obs()
            for i in range(env.max_steps):
                obs, r, d, info = env.step(fn(obs))
                if d:
                    ok += 1
                    times.append(i * DT)
                    break
        res[kind] = (ok / n, float(np.mean(times)) if times else float('nan'))
        print(f'{kind:8s} success {ok}/{n}  mean time {res[kind][1]:.2f}s')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('ckpt')
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    if a.task == 'walk':
        eval_walk(a.ckpt, a.n)
    else:
        eval_getup(a.ckpt, a.n)
