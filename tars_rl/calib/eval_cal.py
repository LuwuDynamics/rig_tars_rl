"""Evaluate the firmware controller (C in the loop) in the ORIGINAL simulator and in the REAL-CALIBRATED simulator (servo fit + command delay + IMU lag).
usage: python calib/eval_cal.py [--n 40] [--only walk|getup]  -> prints a table, writes runs/eval_cal_<tag>.md"""
import argparse
import json
import math
import os
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "rl"))
from tars_env import TarsEnv, DT  # noqa: E402
from c_in_loop import CController  # noqa: E402
from test_c_loop import make_ctl  # noqa: E402
import c_in_loop  # noqa: E402

HIP, LIFT = 2.4, 0.014
R = lambda p: os.path.join(ROOT, p)


def calibrated_kwargs(fit_json=None, lag=1, lag_jitter=1, delay_jitter_ms=4.0, tau_range=(1.15, 1.42), ctrl_dt=None):
    P = json.load(open(fit_json or R("calib/servo_fit_v1.json")))["params"]
    return dict(servo_cal=dict(kp=P["kp"], kd=P["kd"], arm=P["arm"], tau=P["tau"], fric=P["fric"]), cmd_delay_ms=P["delay_ms"], cmd_delay_jitter_ms=delay_jitter_ms,
                imu_lag_samples=lag, imu_lag_jitter=lag_jitter, tau_range=tau_range, ctrl_dt=ctrl_dt)


def run_episode(ctl, seed, T, kind="walk", cmd=(0.0, 0.0), push=0.0, init=None, env_kw=None):
    dt = (env_kw or {}).get("ctrl_dt") or DT
    kw = dict(task="getup" if kind == "getup" else "walk", hip_scale=HIP, lift_scale=LIFT, max_steps=int(T / dt), randomize=True, difficulty=1.0, cmd_fixed=(0, 0),
              no_terminate=True, push_prob_per_s=0.25 if push > 0 else 0.0, push_impulse=(0.0, push))
    if init:
        kw["getup_init"] = init
    if env_kw:
        kw.update(env_kw)
    env = TarsEnv(seed=seed, **kw)
    ctl.reset(2.0)
    fifo = deque(maxlen=64)                                       # IMU samples (acc, gyro), newest last: the controller sees the one `lag` samples old

    def hook(acc, gyr):
        fifo.append((np.array(acc), np.array(gyr)))
        lag = getattr(env, "pr", {}).get("imu_lag", 0) if hasattr(env, "pr") else 0
        a_, g_ = fifo[-1 - min(lag, len(fifo) - 1)]
        ctl.imu(a_, g_, 0.01)
    env.imu_hook = hook
    env.reset()
    ctl.set_kp(env.filter.kp)
    lag = env.pr.get("imu_lag", 0)
    x0, y0 = env.d.qpos[0], env.d.qpos[1]
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    yaw0 = math.atan2(R0[1, 0], R0[0, 0])
    falls, up_time, prev_state, succ_t = 0, 0.0, 0, None
    yaw_prev, yaw_acc = yaw0, 0.0
    n = 0
    st = 0
    for i in range(env.max_steps):
        g_ = fifo[-1 - min(lag, len(fifo) - 1)][1]
        ql, qr, z, st = ctl.step(g_, dt, cmd[0], cmd[1])
        if st == 5 and prev_state != 5:
            falls += 1
        prev_state = st
        obs, r, d, info = env.step(np.array([ql / HIP, qr / HIP, z / LIFT]))
        Rm = env.d.xmat[env.id_torso].reshape(3, 3)
        yaw_now = math.atan2(Rm[1, 0], Rm[0, 0])
        yaw_acc += (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw_now
        if Rm[2, 2] > 0.9:
            up_time += dt
        n += 1
        if kind == "getup" and d:
            succ_t = i * dt
            break
    Rm = env.d.xmat[env.id_torso].reshape(3, 3)
    along = (env.d.qpos[0] - x0) * math.cos(yaw0) + (env.d.qpos[1] - y0) * math.sin(yaw0)
    return dict(falls=falls, dx=float(along), yaw=float(yaw_acc), succ_t=succ_t, fell_finally=bool(Rm[2, 2] < 0.5), upright_frac=up_time / (n * dt + 1e-9))


_CTL = {}


def _init_worker():
    c_in_loop.build = lambda: None                    # the parent already compiled the C controller


def _job(args):
    gait_files, kind, seed, T, cmd, push, init, env_kw, tag = args
    key = (gait_files, tag)
    if key not in _CTL:
        f, b, t, tr, gu = gait_files
        _CTL[key] = make_ctl(f, gu, back_json=b, turn_json=t, turnr_json=tr)
    return run_episode(_CTL[key], seed, T, kind, cmd, push, init, env_kw)


def table(pool, gait_files, env_kw, n, tag, T=15.0, only=None):
    rows = []

    def walk(name, cmd, push):
        res = pool.map(_job, [(gait_files, "walk", 5000 + k, T, cmd, push, None, env_kw, tag) for k in range(n)])
        nf = np.mean([r["falls"] == 0 for r in res]) * 100
        up = np.mean([not r["fell_finally"] for r in res]) * 100
        prog = np.mean([r["dx"] for r in res]) / T * 100
        yaw = np.mean([r["yaw"] for r in res]) / T
        prog_ok = np.mean([r["dx"] for r in res if r["falls"] == 0] or [0.0]) / T * 100
        rows.append((name, "%.0f%%" % nf, "%.0f%%" % up, "%+.1f cm/s" % prog, "%+.1f cm/s" % prog_ok, "%+.2f rad/s" % yaw, "%.2f" % np.mean([r["falls"] for r in res])))
        print(rows[-1], flush=True)
    if only in (None, "walk"):
        walk("forward (cmd +20)", (20.0, 0.0), 0.0)
        walk("forward (cmd +20), pushes <=0.08 N s", (20.0, 0.0), 0.08)
        walk("backward (cmd -20)", (-20.0, 0.0), 0.0)
        walk("turn left (cmd +20)", (0.0, 20.0), 0.0)
        walk("turn right (cmd -20)", (0.0, -20.0), 0.0)
    if only in (None, "getup"):
        for kind in ("back", "front"):
            res = pool.map(_job, [(gait_files, "getup", 6000 + k, 30.0, (0, 0), 0.0, dict(kind=kind, hip=(0, 0)), env_kw, tag) for k in range(n)])
            ok = [r for r in res if r["succ_t"] is not None]
            rows.append(("get-up from %s (<=3 attempts)" % kind, "%d/%d" % (len(ok), n), "-", "%.1f s" % (np.mean([r["succ_t"] for r in ok]) if ok else float("nan")), "-", "-", "-"))
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--fwd", default=R("data/walk_cma_cal/fwd_s5.json"))
    ap.add_argument("--back", default=R("data/walk_cma_cal/back_s5_mirror_lift2_x085.json"))
    ap.add_argument("--turn", default=R("data/walk_cma_cal/turn_c3.json"))
    ap.add_argument("--turnr", default=None)
    ap.add_argument("--getup", default=R("data/getup_cma/back_tau1.0_rate8.0.npy"))
    ap.add_argument("--tag", default="v9")
    ap.add_argument("--only", default=None)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--xml", default=None, help="model variant file in model/ (e.g. tars_e10.xml = feet 10 mm longer front+back)")
    ap.add_argument("--ctrl-hz", type=int, default=50, help="controller rate in the calibrated simulation (50 or 100)")
    a = ap.parse_args()
    gf = (a.fwd, a.back, a.turn, a.turnr, a.getup)
    c_in_loop.build()
    pool = Pool(a.procs, initializer=_init_worker)
    out = []
    if not a.skip_original:
        print("== ORIGINAL simulator ==", flush=True)
        out.append(("ORIGINAL simulator (servo 6.3-8.4 V, 0-8 ms latency, no IMU lag)", table(pool, gf, None, a.n, a.tag + "_orig", only=a.only)))
    print("== REAL-CALIBRATED simulator (servo fit, delay, IMU lag %d +-1) ==" % a.lag, flush=True)
    ekw = calibrated_kwargs(lag=a.lag, ctrl_dt=(0.01 if a.ctrl_hz == 100 else None))
    if a.xml:
        ekw["xml"] = R(os.path.join("model", a.xml))
    out.append(("REAL-CALIBRATED simulator (fitted servo, command delay, IMU lag %d +-1 sample)%s" % (a.lag, (" on model " + a.xml) if a.xml else ""), table(pool, gf, ekw, a.n, a.tag + "_cal%s%s" % ("100" if a.ctrl_hz == 100 else "", a.xml or ""), only=a.only)))
    path = R("runs/eval_cal_%s.md" % a.tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for title, rows in out:
            f.write("### %s   (gaits: %s)\n\nN=%d episodes per row, 15 s (get-up 30 s)\n\n| test | no fall | ends upright | progress (all) | progress (no-fall) | yaw rate | falls/ep |\n|---|---|---|---|---|---|---|\n" % (title, a.tag, a.n))
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
            f.write("\n")
    print(open(path).read())
    pool.close()


if __name__ == "__main__":
    main()
