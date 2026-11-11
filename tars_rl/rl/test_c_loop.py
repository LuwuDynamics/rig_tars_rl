"""Closed-loop tests with the firmware C controller (tars_ctrl.c + C Mahony) driving the MuJoCo robot.

  python rl/test_c_loop.py walk  runs/walk_cma/fwd.json  [--n 20]
  python rl/test_c_loop.py getup data/getup_cma/back_tau1.0_rate8.0.npy [--n 30]
  python rl/test_c_loop.py stress runs/walk_cma/fwd.json data/getup_cma/back_tau1.0_rate8.0.npy
"""
import argparse, json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim')); sys.path.insert(0, os.path.join(ROOT, 'rl'))
from tars_env import TarsEnv, DT
from c_in_loop import CController

HIP, LIFT = 2.4, 0.014


def make_ctl(gait_json, getup_npy, rate=8.0, tend=7.0, back_json=None, turn_json=None, turnr_json=None):
    g = {}
    if gait_json:
        g['F'] = json.load(open(gait_json))['params']
    if back_json:
        g['B'] = json.load(open(back_json))['params']
    if turn_json:
        g['T'] = json.load(open(turn_json))['params']
    if turnr_json:
        g['R'] = json.load(open(turnr_json))['params']
    x = np.load(getup_npy) if getup_npy else np.zeros(28)
    return CController(g, (x, 14, tend, rate))


def run_episode(ctl, seed, T, kind='walk', cmd=(0.0, 0.0), push=0.0, init=None, xml=None, randomize=True, getup_stop=True):
    kw = dict(task='getup' if kind == 'getup' else 'walk', hip_scale=HIP, lift_scale=LIFT, max_steps=int(T / DT), randomize=randomize,
              difficulty=1.0, cmd_fixed=(0, 0), no_terminate=True, push_prob_per_s=0.25 if push > 0 else 0.0, push_impulse=(0.0, push), xml=xml)
    if init:
        kw['getup_init'] = init
    env = TarsEnv(seed=seed, **kw)
    ctl.reset(2.0)
    env.imu_hook = lambda acc, gyr: ctl.imu(acc, gyr, 0.01)   # installed BEFORE reset: the C filter starts from the settled samples
    env.reset()
    ctl.set_kp(env.filter.kp)
    x0, y0 = env.d.qpos[0], env.d.qpos[1]
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    yaw0 = math.atan2(R0[1, 0], R0[0, 0])
    falls, up_time, prev_state, succ_t = 0, 0.0, 0, None
    yaw_prev, yaw_acc = yaw0, 0.0
    states = []
    for i in range(env.max_steps):
        ql, qr, z, st = ctl.step(env.imu_gyro, DT, cmd[0], cmd[1])
        if st == 5 and prev_state != 5:
            falls += 1
        prev_state = st
        obs, r, d, info = env.step(np.array([ql / HIP, qr / HIP, z / LIFT]))
        R = env.d.xmat[env.id_torso].reshape(3, 3)
        yaw_now = math.atan2(R[1, 0], R[0, 0])
        yaw_acc += (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw_now
        if R[2, 2] > 0.9:
            up_time += DT
        if kind == 'getup' and d:
            succ_t = i * DT
            break
        states.append(st)
    R = env.d.xmat[env.id_torso].reshape(3, 3)
    dxw, dyw = env.d.qpos[0] - x0, env.d.qpos[1] - y0
    along = dxw * math.cos(yaw0) + dyw * math.sin(yaw0)          # progress along the initial heading
    return dict(falls=falls, upright_frac=up_time / (len(states) * DT + 1e-9), final_up=float(R[2, 2]), dx=float(along), dist=float(math.hypot(dxw, dyw)), yaw=float(yaw_acc),
                succ_t=succ_t, final_state=CController.STATES[st], fell_finally=bool(R[2, 2] < 0.5))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--T', type=float, default=20.0)
    ap.add_argument('--push', type=float, default=0.0)
    ap.add_argument('--xml', default=None)
    ap.add_argument('--back', default=None)
    ap.add_argument('--turn', default=None)
    ap.add_argument('--turnr', default=None)
    ap.add_argument('--cmd', default='20,0', help='vx,vyaw in servo counts')
    a = ap.parse_args()
    if a.mode == 'walk':
        ctl = make_ctl(a.files[0], a.files[1] if len(a.files) > 1 else None, back_json=a.back, turn_json=a.turn, turnr_json=a.turnr)
        cmd = tuple(float(v) for v in a.cmd.split(','))
        res = [run_episode(ctl, 5000 + k, a.T, 'walk', cmd=cmd, push=a.push, xml=a.xml) for k in range(a.n)]
        print(f'C-in-loop WALK (cmd vx,vyaw={cmd} counts, mean yaw {np.mean([r["yaw"] for r in res]):+.1f} rad/{a.T:.0f}s, {a.T:.0f}s, push<= {a.push} Ns): falls/episode {np.mean([r["falls"] for r in res]):.2f}; '
              f'episodes with no fall {np.mean([r["falls"]==0 for r in res])*100:.0f}%; ends upright {np.mean([not r["fell_finally"] for r in res])*100:.0f}%; '
              f'mean progress along heading {np.mean([r["dx"] for r in res])*100:.0f} cm (no-fall episodes: {np.mean([r["dx"] for r in res if r["falls"]==0] or [0])*100:.0f} cm); upright time {np.mean([r["upright_frac"] for r in res])*100:.0f}%')
    elif a.mode == 'getup':
        ctl = make_ctl(None, a.files[0])
        for kind in ('back', 'front'):
            res = [run_episode(ctl, 6000 + k, 30.0, 'getup', init=dict(kind=kind, hip=(0, 0)), xml=a.xml) for k in range(a.n)]
            ok = [r for r in res if r['succ_t'] is not None]
            print(f'C-in-loop GETUP from {kind:5s}: success {len(ok)}/{a.n}  mean time {np.mean([r["succ_t"] for r in ok]) if ok else float("nan"):.1f}s (incl. wait + retries up to 3)')
