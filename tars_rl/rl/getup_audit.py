"""Is the learned get-up physically plausible and gentle enough for the real robot?  Per-episode peaks of what the hardware would feel, for the learned
policy and for the firmware keyframe get-up, in the same randomised simulation (same seeds).
  hip speed [rad/s]  : the real SCS0009 reaches ~8-11 rad/s unloaded (the supervisor limits the hips to 8)
  servo torque       : peak |actuator force| as a fraction of the (randomised) torque limit
  torso rate [rad/s] : peak angular speed of the body
  impact [m/s]       : peak downward speed of the torso while it touches the floor (landing / falling back)
  contact force [N]  : peak normal force of any floor contact  (robot weighs about 4 N)
  arm slam [rad/s]   : peak arm speed at the moment an arm box touches the floor with force > 2 N

usage: python rl/getup_audit.py --policy runs/getup_ppo_c2/last.pt [--n 60] [--plant cal]
"""
import argparse
import json
import os
import sys
from collections import deque
from multiprocessing import Pool

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('sim', 'rl', 'calib'):
    sys.path.insert(0, os.path.join(ROOT, p))
from tars_env import DT                                    # noqa: E402
from getup_env import GetupEnv, HIP, LIFT                  # noqa: E402
import eval_cal                                            # noqa: E402
import c_in_loop                                           # noqa: E402
from test_c_loop import make_ctl                           # noqa: E402
from evaluate import load_policy                           # noqa: E402

KEYFRAMES = os.path.join(ROOT, 'data/getup_cma/back_tau1.0_rate8.0.npy')
_CTL, _POL = {}, {}


def _init_worker():
    c_in_loop.build = lambda: None


def peaks(env, acc):
    d, m = env.d, env.m
    acc['hip'] = max(acc['hip'], abs(d.qvel[7]), abs(d.qvel[8]))
    acc['tau'] = max(acc['tau'], float(np.max(np.abs(d.actuator_force[:3]) / np.abs(m.actuator_forcerange[:3, 1]))))
    acc['w'] = max(acc['w'], float(np.linalg.norm(d.qvel[3:6])))
    f6 = np.zeros(6)
    fmax, touching = 0.0, False
    for k in range(d.ncon):
        c = d.contact[k]
        g = {c.geom1, c.geom2}
        if env.g_floor in g:
            mujoco.mj_contactForce(m, d, k, f6)
            fmax = max(fmax, f6[0])
            if env.g_boxes[0] in g:
                touching = True
            if (env.g_boxes[1] in g or env.g_boxes[2] in g) and f6[0] > 2.0:
                acc['slam'] = max(acc['slam'], abs(d.qvel[7]), abs(d.qvel[8]))
    acc['force'] = max(acc['force'], fmax)
    if touching:
        acc['impact'] = max(acc['impact'], -float(d.qvel[2]))


def job(args):
    who, plant, kind, seed, T = args
    init = None if kind == 'random' else dict(kind=kind, hip=(0.0, 0.0))
    cal = eval_cal.calibrated_kwargs(lag=1, lag_jitter=1)
    cal.pop('ctrl_dt', None)
    acc = dict(hip=0.0, tau=0.0, w=0.0, force=0.0, impact=0.0, slam=0.0)
    mix = 1.0 if plant == 'cal' else 0.0
    if who == 'keyframes':
        if 'k' not in _CTL:
            _CTL['k'] = make_ctl(None, KEYFRAMES)
        ctl = _CTL['k']
        env = GetupEnv(seed=seed, plant_cal=cal, plant_mix=mix, continue_after_success=False, max_steps=int(round(T / DT)), action_repeat=1, rate_hip=1e3, rate_lift=1e3, getup_init=init)
        ctl.reset(2.0)
        fifo = deque(maxlen=64)

        def hook(a_, g_):
            fifo.append((np.array(a_), np.array(g_)))
            lag = env.pr.get('imu_lag', 0) if hasattr(env, 'pr') else 0
            aa, gg = fifo[-1 - min(lag, len(fifo) - 1)]
            ctl.imu(aa, gg, 0.01)
        env.imu_hook = hook
        env.reset()
        ctl.set_kp(env.filter.kp)
        lag = env.pr.get('imu_lag', 0)
        for i in range(env.max_steps):
            ql, qr, z, st = ctl.step(fifo[-1 - min(lag, len(fifo) - 1)][1], DT, 0.0, 0.0)
            obs, r, d, info = env.step(np.array([ql / HIP, qr / HIP, z / LIFT]))
            peaks(env, acc)
            if d:
                break
        return env.succeeded, env.t_success, acc
    if who not in _POL:
        _POL[who] = load_policy(who)
    fn, ck = _POL[who]
    a = ck['args']
    kw = dict(seed=seed, plant_cal=cal, plant_mix=mix, continue_after_success=False, max_steps=int(round(T / DT)), action_repeat=1, getup_init=init,
              time_obs=bool(a.get('time_obs', 1)), rate_hip=a.get('rate_hip', 8.0), rate_lift=a.get('rate_lift', 0.25))
    kw.update({k: v for k, v in json.loads(a.get('env_kw') or '{}').items() if k in ('obs_q_mode', 'imu_clip')})
    env = GetupEnv(**kw)
    obs, _ = env.reset()
    done = False
    for i in range(int(round(T / DT)) // 5):
        act = fn(obs)
        for _ in range(5):
            obs, r, done, info = env.step(act)
            peaks(env, acc)
            if done or info['truncated']:
                break
        if done or info['truncated']:
            break
    return env.succeeded, env.t_success, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', action='append', default=[])
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--T', type=float, default=20.0)
    ap.add_argument('--plant', default='cal')
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--kinds', default='back,front')
    a = ap.parse_args()
    c_in_loop.build()
    pool = Pool(a.procs, initializer=_init_worker)
    print('| start | controller | success | hip speed p50 / p95 / max [rad/s] | servo torque p95 | torso rate p95 | impact p95 [m/s] | floor force p95 / max [N] | arm slam p95 [rad/s] |')
    print('|---|---|---|---|---|---|---|---|---|')
    for kind in a.kinds.split(','):
        for who in ['keyframes'] + a.policy:
            res = pool.map(job, [(who, a.plant, kind, 6000 + k, a.T) for k in range(a.n)])
            ok = sum(1 for r in res if r[0])
            P = lambda key, q: np.percentile([r[2][key] for r in res], q)
            name = 'keyframes' if who == 'keyframes' else os.path.basename(os.path.dirname(who)) + '/' + os.path.basename(who)
            print('| %s | %s | %d/%d | %.1f / %.1f / %.1f | %.2f | %.1f | %.2f | %.1f / %.1f | %.1f |' % (
                kind, name, ok, a.n, P('hip', 50), P('hip', 95), P('hip', 100), P('tau', 95), P('w', 95), P('impact', 95), P('force', 95), P('force', 100), P('slam', 95)), flush=True)
    pool.close()


if __name__ == '__main__':
    main()
