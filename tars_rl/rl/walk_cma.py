"""CMA-ES search for a robust *feedback* gait: periodic hip/lift pattern + IMU pitch feedback.
Gradient-free policy search, evaluated in the randomised simulator (friction, CoM, torque, latency, IMU noise ...).

usage: python rl/walk_cma.py fwd --gens 150 --pop 24 --procs 9
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

# parameter vector layout / ranges  (normalised x in [-1,1] -> physical)
NAMES = ['q0', 'a1', 'b1', 'a2', 'b2', 'z0', 'c1', 'd1', 'c2', 'd2', 'f', 'kp_h', 'kd_h', 'kp_z', 'kd_z']
LO = np.array([-0.10, -0.30, -0.30, -0.15, -0.15, -0.006, -0.012, -0.012, -0.006, -0.006, 0.6, -4.0, -1.0, -0.10, -0.03])
HI = np.array([0.10, 0.30, 0.30, 0.15, 0.15, 0.006, 0.012, 0.012, 0.006, 0.006, 2.0, 4.0, 1.0, 0.10, 0.03])


def decode(x):
    return LO + (np.clip(x, -1, 1) + 1) * 0.5 * (HI - LO)


def controller(p, t, up, gyro_y):
    q0, a1, b1, a2, b2, z0, c1, d1, c2, d2, f, kp_h, kd_h, kp_z, kd_z = p
    ph = 2 * math.pi * f * t
    q = q0 + a1 * math.cos(ph) + b1 * math.sin(ph) + a2 * math.cos(2 * ph) + b2 * math.sin(2 * ph)
    z = z0 + c1 * math.cos(ph) + d1 * math.sin(ph) + c2 * math.cos(2 * ph) + d2 * math.sin(2 * ph)
    theta = -math.asin(max(-1.0, min(1.0, up[0])))          # >0: torso pitched forward (nose down)
    q += kp_h * theta + kd_h * gyro_y
    z += kp_z * theta + kd_z * gyro_y
    return q, z


def rollout(args):
    x, seed, T, randomize, ramp, push, mode = args
    p = decode(x)
    env = TarsEnv(seed=seed, task='walk', max_steps=int(T / DT), push_prob_per_s=0.3 if push else 0.0, push_impulse=(0.0, push), cmd_fixed=(0, 0),
                  randomize=randomize, difficulty=1.0)
    obs, _ = env.reset()
    x0, y0 = env.d.qpos[0], env.d.qpos[1]
    yaw0 = math.atan2(env.d.xmat[env.id_torso][3], env.d.xmat[env.id_torso][0])
    fell = False
    yaw_acc, yaw_prev = 0.0, yaw0
    for i in range(env.max_steps):
        t = i * DT
        g = min(1.0, t / ramp) if ramp > 0 else 1.0           # ease the pattern in so start-up is not a shock
        up = env.filter.up_body()
        q, z = controller(p, t, up, env.imu_gyro[1])
        q, z = q * g, z * g
        qr = -q if mode == 'turn' else q
        a = np.array([q / env.hip_scale, qr / env.hip_scale, z / env.lift_scale])
        obs, r, d, info = env.step(a)
        Rn = env.d.xmat[env.id_torso].reshape(3, 3)
        yaw_now = math.atan2(Rn[1, 0], Rn[0, 0])
        yaw_acc += (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi       # accumulate wrapped increments
        yaw_prev = yaw_now
        if d:
            fell = True
            break
    R = env.d.xmat[env.id_torso].reshape(3, 3)
    dx = (env.d.qpos[0] - x0) * math.cos(yaw0) + (env.d.qpos[1] - y0) * math.sin(yaw0)
    lat = -(env.d.qpos[0] - x0) * math.sin(yaw0) + (env.d.qpos[1] - y0) * math.cos(yaw0)
    dyaw = yaw_acc
    return fell, dx, lat, (i + 1) * DT, dyaw, math.hypot(env.d.qpos[0] - x0, env.d.qpos[1] - y0)


T_REF = 10.0
FALL_PEN = 0.6
CAP = 0.45          # progress per 10 s that earns full reward (0.45 m / 10 s = 4.5 cm/s)
MODE = 'walk'


def objective(res, sign):
    fell = np.array([r[0] for r in res])
    surv = np.array([r[3] for r in res])
    if MODE == 'turn':
        prog = np.array([r[4] for r in res]) * sign / (T_REF / 10.0)          # rad per 10 s (>0 = left)
        drift = np.array([r[5] for r in res]) / (T_REF / 10.0)
        score = np.where(fell, -FALL_PEN * (1 - 0.5 * surv / T_REF), np.clip(prog / 8.0, -0.1, 0.45) - 0.5 * drift)
        return float(score.mean()), float(fell.mean()), float(np.mean(prog[~fell])) if (~fell).any() else 0.0
    dx = np.array([r[1] for r in res]) * sign
    lat = np.array([abs(r[2]) for r in res])
    score = np.where(fell, -FALL_PEN * (1 - 0.5 * surv / T_REF), np.clip(dx / (T_REF / 10.0), -0.1, CAP) - 0.3 * lat / (T_REF / 10.0))
    return float(score.mean()), float(fell.mean()), float(dx[~fell].mean()) if (~fell).any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', choices=['fwd', 'back', 'turn', 'turnr'])
    ap.add_argument('--gens', type=int, default=150)
    ap.add_argument('--pop', type=int, default=24)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--nseeds', type=int, default=6)
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--out', default='runs/walk_cma')
    ap.add_argument('--push', type=float, default=0.0, help='max random push impulse [N s] (0.3 pushes/s)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--x0', default='', help='start the search from this saved solution (.npy)')
    ap.add_argument('--sigma', type=float, default=0.35)
    ap.add_argument('--fall_pen', type=float, default=0.6)
    ap.add_argument('--cap', type=float, default=0.45)
    a = ap.parse_args()
    global T_REF, FALL_PEN, CAP, MODE
    T_REF = a.T
    FALL_PEN, CAP = a.fall_pen, a.cap
    MODE = 'turn' if a.dir in ('turn', 'turnr') else 'walk'
    sign = -1.0 if a.dir in ('back', 'turnr') else 1.0
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    x0 = np.zeros(len(NAMES))
    x0[NAMES.index('f')] = (1.25 - 0.6) / 1.4 * 2 - 1
    if a.x0:
        x0 = np.load(os.path.join(ROOT, a.x0))
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {'popsize': a.pop, 'seed': 3, 'verbose': -9, 'bounds': [-1, 1]})
    pool = Pool(a.procs)
    best = (-1e9, None, None)
    t0 = time.time()
    for g in range(a.gens):
        xs = es.ask()
        jobs = [(x, 10_000 * g + k, a.T, True, 1.0, a.push, MODE) for x in xs for k in range(a.nseeds)]
        flat = pool.map(rollout, jobs)
        scores, falls, dxs = [], [], []
        for i in range(len(xs)):
            s, f, d = objective(flat[i * a.nseeds:(i + 1) * a.nseeds], sign)
            scores.append(s); falls.append(f); dxs.append(d)
        es.tell(xs, (-np.array(scores)).tolist())
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (scores[i], xs[i].copy(), (falls[i], dxs[i]))
            np.save(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}_ckpt.npy'), best[1])
        if g % 5 == 0 or g == a.gens - 1:
            print(f'gen {g:3d} best {max(scores):6.3f} (all-time {best[0]:6.3f}; fall {best[2][0]:.2f} dx {best[2][1]*1000:5.0f} mm)  mean fall {np.mean(falls):.2f}  {time.time()-t0:.0f}s', flush=True)
    xb = best[1]
    np.save(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}.npy'), xb)
    p = decode(xb)
    print('best params:', {k: round(float(v), 4) for k, v in zip(NAMES, p)})
    # validate on fresh seeds + no-DR nominal
    res = pool.map(rollout, [(xb, 777_000 + k, a.T, True, 1.0, a.push, MODE) for k in range(60)])
    s, f, d = objective(res, sign)
    print(f'VALIDATION (60 fresh randomised seeds, {a.T:.0f} s episodes): fall rate {f:.2f}, mean progress of survivors {d:.3f} ({"rad" if MODE=="turn" else "m"} per 10 s)')
    json.dump(dict(params={k: float(v) for k, v in zip(NAMES, p)}, fall=f, dx=d), open(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}.json'), 'w'), indent=1)
    pool.close()


if __name__ == '__main__':
    main()
