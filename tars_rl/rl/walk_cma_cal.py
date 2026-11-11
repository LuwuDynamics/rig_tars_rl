"""CMA-ES search for a robust feedback gait in the REAL-CALIBRATED simulator (servo fit + command delay + IMU lag, fitted to the 2026-10-28 real data).
Same gait family as rl/walk_cma.py (15 parameters), but the objective also penalises the shaking (>5 Hz energy in the hip command) and the servo effort.
The original rl/walk_cma.py and its results (runs/walk_cma) are untouched.

usage: python rl/walk_cma_cal.py fwd --gens 60 --pop 28 --procs 9 --tag _c1 --x0 data/walk_cma/fwd_v4.npy --sigma 0.15
"""
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
from tars_env import TarsEnv, DT, COUNT_RAD            # noqa: E402
import walk_cma as W                                     # noqa: E402   (parameter layout, controller)
from eval_cal import calibrated_kwargs                   # noqa: E402

ENV_KW = {}
W_SMOOTH = 0.0         # penalty weight: (body rate rms [deg/s] / 100 + vertical acceleration rms [m/s^2] / 5): the shaking the eye sees
W_HF = 0.15            # penalty weight: rms of the >5 Hz part of the hip command [counts] / 10
W_EFF = 0.30           # penalty weight: mean squared normalised actuator force


def hf_rms(cmd, dt=DT):
    """rms [counts] of the hip command above ~5 Hz (command minus its 0.2 s moving average)"""
    c = np.asarray(cmd, float)
    k = max(3, int(round(0.2 / dt)))
    if len(c) < 3 * k:
        return 0.0
    ker = np.ones(k) / k
    base = np.convolve(c, ker, mode='same')
    return float(np.std((c - base)[k:-k]))


def rollout(args):
    x, seed, T, push, mode, ramp = args
    p = W.decode(x)
    dt = ENV_KW.get('ctrl_dt') or DT
    env = TarsEnv(seed=seed, task='walk', max_steps=int(T / dt), push_prob_per_s=0.3 if push else 0.0, push_impulse=(0.0, push), cmd_fixed=(0, 0),
                  randomize=True, difficulty=1.0, **ENV_KW)
    env.reset()
    x0, y0 = env.d.qpos[0], env.d.qpos[1]
    yaw0 = math.atan2(env.d.xmat[env.id_torso][3], env.d.xmat[env.id_torso][0])
    fell = False
    yaw_acc, yaw_prev = 0.0, yaw0
    cmds, eff = [], 0.0
    sm_g, sm_a = [], []                                          # body rate (x, y) and vertical acceleration of the torso: what the eye sees as shaking
    for i in range(env.max_steps):
        t = i * dt
        g = min(1.0, t / ramp) if ramp > 0 else 1.0
        q, z = W.controller(p, t, env.up_ctrl(), env.gyro_ctrl()[1])
        q, z = q * g, z * g
        q = max(-0.6, min(0.6, q))
        z = max(-0.014, min(0.014, z))
        qr = -q if mode == 'turn' else q
        a = np.array([q / env.hip_scale, qr / env.hip_scale, z / env.lift_scale])
        cmds.append(q / COUNT_RAD)
        obs, r, d, info = env.step(a)
        qf = env.d.actuator_force[:3] / env.tau_nom
        eff += float(qf @ qf)
        if t >= 1.0:
            sd = env.d.sensordata
            sm_g.append(float(sd[0] ** 2 + sd[1] ** 2))
            sm_a.append(float(sd[5]))
        Rn = env.d.xmat[env.id_torso].reshape(3, 3)
        yaw_now = math.atan2(Rn[1, 0], Rn[0, 0])
        yaw_acc += (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw_now
        if d:
            fell = True
            break
    n = i + 1
    dx = (env.d.qpos[0] - x0) * math.cos(yaw0) + (env.d.qpos[1] - y0) * math.sin(yaw0)
    lat = -(env.d.qpos[0] - x0) * math.sin(yaw0) + (env.d.qpos[1] - y0) * math.cos(yaw0)
    smg = math.degrees(math.sqrt(float(np.mean(sm_g)))) if sm_g else 0.0       # deg/s rms (roll + pitch rate)
    sma = float(np.std(sm_a)) if sm_a else 0.0                                 # m/s^2 rms (bounce)
    return fell, dx, lat, n * dt, yaw_acc, math.hypot(env.d.qpos[0] - x0, env.d.qpos[1] - y0), hf_rms(cmds, dt), eff / n, smg, sma


T_REF, FALL_PEN, CAP, MODE = 15.0, 0.8, 0.30, 'walk'
STILL_PEN, STILL_FRAC, FALL_CREDIT = 0.25, 0.15, 0.4
TURN_CAP = 0.45          # turning: progress per 10 s that earns full reward is 8 x TURN_CAP rad (0.45 -> 3.6 rad / 10 s = 0.36 rad/s)


def objective(res, sign):
    fell = np.array([r[0] for r in res])
    surv = np.array([r[3] for r in res])
    hf = np.array([r[6] for r in res])
    eff = np.array([r[7] for r in res])
    smg = np.array([r[8] for r in res])
    sma = np.array([r[9] for r in res])
    pen = W_HF * hf / 10.0 + W_EFF * eff + W_SMOOTH * (smg / 100.0 + sma / 5.0)
    if MODE == 'turn':
        prog = np.array([r[4] for r in res]) * sign / (T_REF / 10.0)
        drift = np.array([r[5] for r in res]) / (T_REF / 10.0)
        still = (~fell) & (prog / 8.0 < 0.1)
        score = np.where(fell, -FALL_PEN * (1 - 0.5 * surv / T_REF), np.clip(prog / 8.0, -0.1, TURN_CAP) - 0.5 * drift) - pen - STILL_PEN * still
        return float(score.mean()), float(fell.mean()), float(np.mean(prog[~fell])) if (~fell).any() else 0.0, float(hf.mean())
    dx = np.array([r[1] for r in res]) * sign
    lat = np.array([abs(r[2]) for r in res])
    dxn = dx / (T_REF / 10.0)
    still = (~fell) & (dxn < STILL_FRAC * CAP)            # a gait that does not walk is not a solution (the search otherwise collapses to standing still)
    speed = dx / np.maximum(surv, 1.0)                    # m/s up to the fall
    credit = FALL_CREDIT * np.clip(speed / (CAP / 10.0), 0.0, 1.0)    # walking well before a fall is better than not walking (gives the search a gradient)
    score = np.where(fell, -FALL_PEN * (1 - 0.5 * surv / T_REF) + credit, np.clip(dxn, -0.1, CAP) - 0.3 * lat / (T_REF / 10.0)) - pen - STILL_PEN * still
    return float(score.mean()), float(fell.mean()), float(dx[~fell].mean()) if (~fell).any() else 0.0, float(hf.mean())


def init_worker(envkw, lohi=None):
    ENV_KW.update(envkw)
    if lohi is not None:                       # spawned workers import the module afresh: they must decode with the main process' (narrowed) --fmax/--fmin ranges
        W.LO[:], W.HI[:] = lohi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', choices=['fwd', 'back', 'turn', 'turnr'])
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--pop', type=int, default=28)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--nseeds', type=int, default=20)
    ap.add_argument('--T', type=float, default=15.0)
    ap.add_argument('--out', default='runs/walk_cma_cal')
    ap.add_argument('--push', type=float, default=0.06)
    ap.add_argument('--tag', default='_c1')
    ap.add_argument('--x0', default='')
    ap.add_argument('--sigma', type=float, default=0.15)
    ap.add_argument('--fall_pen', type=float, default=0.8)
    ap.add_argument('--cap', type=float, default=0.30)
    ap.add_argument('--w_hf', type=float, default=0.15)
    ap.add_argument('--w_eff', type=float, default=0.30)
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--lag_jitter', type=int, default=1)
    ap.add_argument('--turn_cap', type=float, default=0.45)
    ap.add_argument('--w_smooth', type=float, default=0.0, help='weight of the body shaking penalty (body rate rms / 100 deg/s + bounce rms / 5 m/s^2)')
    ap.add_argument('--fmax', type=float, default=0.0, help='upper limit of the gait frequency [Hz] (0 = the original 2.0)')
    ap.add_argument('--fmin', type=float, default=0.0, help='lower limit of the gait frequency [Hz] (0 = the original 0.6)')
    ap.add_argument('--xml', default='', help='model variant in model/ (e.g. tars_p10.xml: 10 mm foot plates front+back)')
    ap.add_argument('--zero', default='', help='comma separated parameters held at 0 (e.g. a2,b2: a hip pattern without second harmonic = a plain sinusoid)')
    ap.add_argument('--ctrl_hz', type=int, default=50, help='controller rate in the simulation: 50 (original) or 100 (firmware option)')
    a = ap.parse_args()
    global T_REF, FALL_PEN, CAP, MODE, W_HF, W_EFF, TURN_CAP, W_SMOOTH
    T_REF, FALL_PEN, CAP, W_HF, W_EFF, TURN_CAP, W_SMOOTH = a.T, a.fall_pen, a.cap, a.w_hf, a.w_eff, a.turn_cap, a.w_smooth
    MODE = 'turn' if a.dir in ('turn', 'turnr') else 'walk'
    sign = -1.0 if a.dir in ('back', 'turnr') else 1.0
    envkw = calibrated_kwargs(lag=a.lag, lag_jitter=a.lag_jitter)
    if a.ctrl_hz == 100:
        envkw['ctrl_dt'] = 0.01
    if a.xml:
        envkw['xml'] = os.path.join(ROOT, 'model', a.xml)
    ENV_KW.update(envkw)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    x0 = np.zeros(len(W.NAMES))
    x0[W.NAMES.index('f')] = (1.25 - 0.6) / 1.4 * 2 - 1
    if a.x0:
        x0 = np.load(os.path.join(ROOT, a.x0))
    if a.zero:                                                     # frozen parameters: lo = hi = 0 (the workers get the ranges through the Pool initializer)
        for nm in a.zero.split(','):
            W.LO[W.NAMES.index(nm)] = 0.0
            W.HI[W.NAMES.index(nm)] = 0.0
    if a.fmax or a.fmin:                                            # narrower frequency range: the start point keeps its parameters (f clipped into the range)
        p0 = W.decode(x0)
        fi = W.NAMES.index('f')
        if a.fmin:
            W.LO[fi] = a.fmin
        if a.fmax:
            W.HI[fi] = a.fmax
        p0[fi] = min(max(p0[fi], W.LO[fi]), W.HI[fi])
        x0 = np.clip((p0 - W.LO) / np.where(W.HI > W.LO, W.HI - W.LO, 1.0) * 2 - 1, -1, 1)
        print(f'gait frequency limited to {W.LO[fi]:.2f}..{W.HI[fi]:.2f} Hz; start point f = {p0[fi]:.2f} Hz', flush=True)
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {'popsize': a.pop, 'seed': 3, 'verbose': -9, 'bounds': [-1, 1]})
    pool = Pool(a.procs, initializer=init_worker, initargs=(envkw, (W.LO.copy(), W.HI.copy())))
    # reference: the start point in the calibrated simulator (same fresh seeds as the validation)
    ref = pool.map(rollout, [(x0, 888_000 + k, a.T, a.push, MODE, 1.0) for k in range(60)])
    s0, f0, d0, h0 = objective(ref, sign)
    print(f'START POINT in the calibrated simulator: score {s0:.3f}  fall rate {f0:.2f}  progress {d0:.3f} m/15 s  shaking {h0:.1f} counts rms  body rate {np.mean([r[8] for r in ref]):.0f} deg/s  bounce {np.mean([r[9] for r in ref]):.1f} m/s^2', flush=True)
    best = (-1e9, None, None)
    t0 = time.time()
    for g in range(a.gens):
        xs = es.ask()
        jobs = [(x, 10_000 * g + k, a.T, a.push, MODE, 1.0) for x in xs for k in range(a.nseeds)]
        flat = pool.map(rollout, jobs)
        sc, fl, dxs, hfs = [], [], [], []
        for i in range(len(xs)):
            s, f, d, h = objective(flat[i * a.nseeds:(i + 1) * a.nseeds], sign)
            sc.append(s); fl.append(f); dxs.append(d); hfs.append(h)
        es.tell(xs, (-np.array(sc)).tolist())
        i = int(np.argmax(sc))
        if sc[i] > best[0]:
            best = (sc[i], xs[i].copy(), (fl[i], dxs[i], hfs[i]))
            np.save(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}_ckpt.npy'), best[1])
        if g % 5 == 0 or g == a.gens - 1:
            print(f'gen {g:3d} best {max(sc):6.3f} (all-time {best[0]:6.3f}; fall {best[2][0]:.2f} dx {best[2][1]*1000:5.0f} mm shake {best[2][2]:4.1f})  mean fall {np.mean(fl):.2f}  {time.time()-t0:.0f}s', flush=True)
    xb = best[1]
    np.save(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}.npy'), xb)
    p = W.decode(xb)
    print('best params:', {k: round(float(v), 4) for k, v in zip(W.NAMES, p)})
    res = pool.map(rollout, [(xb, 777_000 + k, a.T, a.push, MODE, 1.0) for k in range(60)])
    s, f, d, h = objective(res, sign)
    print(f'VALIDATION (60 fresh randomised seeds, calibrated simulator, {a.T:.0f} s episodes): fall rate {f:.2f}, progress of survivors {d:.3f} ({"rad/10 s" if MODE=="turn" else "m per episode"}), shaking {h:.1f} counts rms, body rate {np.mean([r[8] for r in res]):.0f} deg/s, bounce {np.mean([r[9] for r in res]):.1f} m/s^2'
          f'   [start point was: fall {f0:.2f}, progress {d0:.3f}, shaking {h0:.1f}, body rate {np.mean([r[8] for r in ref]):.0f}, bounce {np.mean([r[9] for r in ref]):.1f}]')
    json.dump(dict(params={k: float(v) for k, v in zip(W.NAMES, p)}, fall=f, dx=d, shake=h, start=dict(fall=f0, dx=d0, shake=h0),
                   env=dict(lag=a.lag, lag_jitter=a.lag_jitter, ctrl_hz=a.ctrl_hz, xml=a.xml), cap=a.cap, fall_pen=a.fall_pen, w_hf=a.w_hf, w_eff=a.w_eff), open(os.path.join(ROOT, a.out, f'{a.dir}{a.tag}.json'), 'w'), indent=1)
    pool.close()


if __name__ == '__main__':
    main()
