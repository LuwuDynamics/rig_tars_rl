"""The show motions that are in the firmware (nod, bow, headbang and the emotions), played by the *firmware supervisor* (tars_ctrl.c MOTION state + the generated tables,
C code in the loop) in the calibrated simulator - not by the Python functions.  Checks what the host tests cannot: that the whole chain works on the simulated robot.

usage: python rl/fun_motions_c.py check [--n 30]      randomised plant: every motion of the firmware's table at full and at half speed must go STAND -> MOTION -> STAND, upright, no fall
       python rl/fun_motions_c.py abort [--n 10]      cut short by `stop`, by a walking command, a request while walking, a request that never gets its stand,
                                                      a request while lying down, and a poke during the bow (-> MOTION -> FALLEN -> learned get-up -> STAND)
       python rl/fun_motions_c.py render              media/fun/fw_*.mp4: the firmware bow, and bow + poke -> fall -> get-up
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fun_sim as fs                                   # noqa: E402
import fun_ctl                                         # noqa: E402
from fun_sim import DT                                 # noqa: E402

ST = {n: i for i, n in enumerate(fun_ctl.STATE_NAMES)}


def motion_names():
    return fun_ctl.get_ctl().motion_names()          # every motion in the firmware's table (nod, bow, headbang + the emotions)


OUT = os.path.join(fs.ROOT, 'media', 'fun')


def pitch(env):
    return fs.torso_pitch_deg(env)


def run_motion(name, speed=1.0, seed=0, randomize=True, t_req=1.0, stop_at=None, cmd_at=None, poke_at=None, T_extra=6.0):
    """request `name` at t_req and run until the supervisor is back in STAND (or time is up); returns a dict of what happened"""
    r = fun_ctl.CtlRun(seed=seed, randomize=randomize, T=60.0)
    ctl, env = r.ctl, r.env
    length = ctl.motion_length(name) / speed
    states, last = [], None
    p_min = p_max = 0.0
    t_start = t_end = None
    asked = None
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    fwd = R0[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    n_max = int((t_req + length + T_extra + 20.0) / DT)
    for i in range(n_max):
        t = r.t
        if asked is None and t >= t_req:
            asked = ctl.motion_start(name, speed)
        vx = 23.0 if (cmd_at is not None and t >= cmd_at) else 0.0
        push = None
        if poke_at is not None and abs(t - poke_at) < DT / 2:
            push = (-fwd[0] * 8.0, -fwd[1] * 8.0)
        if stop_at is not None and abs(t - stop_at) < DT / 2:
            ctl.motion_stop()
        st = r.step(vx, 0.0, push=push)
        if st != last:
            states.append((round(t, 2), fun_ctl.STATE_NAMES[st]))
            last = st
        if st == ST['MOTION']:
            t_start = t_start if t_start is not None else t
            p = pitch(env)
            p_min, p_max = min(p_min, p), max(p_max, p)
        if t_start is not None and st == ST['STAND'] and t_end is None:
            t_end = t
        if t_end is not None and t > t_end + 1.0 and (cmd_at is None or t > cmd_at + 3.0):
            break
        if t_end is not None and push is None and cmd_at is None and t > t_end + 1.0:
            break
    seq = [s for _, s in states]
    return dict(r=r, asked=asked, states=states, seq=seq, t_start=t_start, t_end=t_end, duration=(t_end - t_start) if (t_start is not None and t_end is not None) else None,
                length=length, p_min=p_min, p_max=p_max, end=ctl.motion_end(), upright=r.upright(), tilt=fs.torso_tilt_deg(env), falls=r.stats['falls'], max_tilt=r.stats['max_tilt'])


def cmd_check(n):
    print('%-9s %5s %5s %8s %8s %12s %7s' % ('motion', 'speed', 'ok', 'duration', 'expected', 'pitch range', 'max tilt'))
    bad = 0
    for nm in motion_names():
        for sp in (1.0, 0.5):
            oks, durs, pmin, pmax, tilts = 0, [], [], [], []
            exp = None
            for seed in range(n):
                c = run_motion(nm, sp, seed=100 + seed)
                exp = c['length']
                good = (c['asked'] == 0 and c['seq'] == ['STAND', 'MOTION', 'STAND'] and c['end'] == 1 and c['upright'] and c['falls'] == 0
                        and c['duration'] is not None and abs(c['duration'] - c['length']) < 0.5)
                oks += good
                if c['duration'] is not None:
                    durs.append(c['duration'])
                pmin.append(c['p_min']); pmax.append(c['p_max']); tilts.append(c['max_tilt'])
                if not good:
                    bad += 1
                    print('   FAIL', nm, sp, 'seed', 100 + seed, c['asked'], c['states'][:8], 'end', c['end'], 'upright', c['upright'])
            print('%-9s %5.1f %2d/%-2d %7.2fs %7.2fs [%4.0f,%4.0f]° %6.1f°' % (nm, sp, oks, n, np.mean(durs) if durs else float('nan'), exp, min(pmin), max(pmax), max(tilts)), flush=True)
    return bad


def expect(ok, text):
    print('  %-4s %s' % ('ok' if ok else 'FAIL', text), flush=True)
    return 0 if ok else 1


def cmd_abort(n):
    bad = 0
    print('stop in the middle of the bow (full / half speed), hips go home first, then the lift:')
    for sp in (1.0, 0.5):
        res = [run_motion('bow', sp, seed=200 + k, t_req=1.0, stop_at=1.0 + 3.0 / sp) for k in range(n)]
        bad += expect(all(c['seq'] == ['STAND', 'MOTION', 'STAND'] and c['end'] == 2 and c['upright'] and c['falls'] == 0 for c in res),
                      'speed %.1f: %d/%d STAND -> MOTION -> STAND, end=stop, upright, no fall; back to STAND in %.2f..%.2f s' % (
                          sp, sum(c['seq'] == ['STAND', 'MOTION', 'STAND'] and c['end'] == 2 for c in res), n, min(c['duration'] for c in res) - 3.0 / sp, max(c['duration'] for c in res) - 3.0 / sp))
    print('a walking command in the middle of the headbang (the motion ends, the robot walks):')
    res = [run_motion('headbang', 1.0, seed=300 + k, t_req=1.0, cmd_at=3.5) for k in range(n)]
    bad += expect(all(c['end'] == 3 and c['falls'] == 0 and c['seq'][:3] == ['STAND', 'MOTION', 'STAND'] for c in res),
                  '%d/%d ended with end=walking command, then %s' % (sum(c['end'] == 3 for c in res), n, res[0]['seq'][2:5]))
    print('a request made while the robot is walking waits for it to stand:')
    r = fun_ctl.CtlRun(seed=400, randomize=True, T=60.0)
    for _ in range(int(0.8 / DT)):
        r.step(0, 0)
    for _ in range(int(3.0 / DT)):
        r.step(23.0, 0)
    asked = r.ctl.motion_start('nod', 1.0)
    seq, last = [], None
    for _ in range(int(14.0 / DT)):
        st = r.step(0.0, 0.0)
        if st != last:
            seq.append(fun_ctl.STATE_NAMES[st]); last = st
    bad += expect(asked == 0 and seq == ['WALK_F', 'STAND', 'MOTION', 'STAND'] and r.ctl.motion_end() == 1 and r.upright(), 'accepted while WALK_F, then %s, end=%d' % (seq, r.ctl.motion_end()))
    print('a request whose walking command never stops times out after 3 s:')
    r = fun_ctl.CtlRun(seed=401, randomize=True, T=60.0)
    for _ in range(int(0.8 / DT)):
        r.step(0, 0)
    for _ in range(int(2.0 / DT)):
        r.step(23.0, 0)
    asked = r.ctl.motion_start('nod', 1.0)
    seq, last = [], None
    for _ in range(int(5.0 / DT)):
        st = r.step(23.0, 0.0)
        if st != last:
            seq.append(fun_ctl.STATE_NAMES[st]); last = st
    bad += expect(asked == 0 and 'MOTION' not in seq and r.ctl.motion_end() == 5, 'accepted, never started (%s), end=%d (timed out)' % (seq, r.ctl.motion_end()))
    print('a request while lying down is refused:')
    r = fun_ctl.CtlRun(seed=402, randomize=True, T=60.0)
    env = r.env
    for _ in range(int(0.8 / DT)):
        r.step(0, 0)
    R0 = env.d.xmat[env.id_torso].reshape(3, 3)
    fwd = R0[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    r.step(0, 0, push=(-fwd[0] * 8.0, -fwd[1] * 8.0))
    st = 0
    for _ in range(int(3.0 / DT)):
        st = r.step(0, 0)
        if st == ST['FALLEN']:
            break
    asked = r.ctl.motion_start('nod', 1.0)
    bad += expect(st == ST['FALLEN'] and asked == -3, 'state %s: motion_start -> %d (busy)' % (fun_ctl.STATE_NAMES[st], asked))
    print('a poke during the bow (hard enough to topple it): the tilt limit / the leg guard end the motion as a fall, the learned get-up stands it up again:')
    res = [run_motion('bow', 1.0, seed=500 + k, t_req=1.0, poke_at=1.0 + 2.8, T_extra=12.0) for k in range(n)]
    fell = [c for c in res if c['end'] in (4, 6)]                 # 4 = the tilt limit, 6 = the leg guard (the legs tilt before the torso does)
    good = [c for c in fell if 'FALLEN' in c['seq'] and 'GETUP_NN' in c['seq'] and c['seq'][-1] == 'STAND' and c['upright']]
    bad += expect(len(fell) >= n // 2 and len(good) == len(fell), '%d/%d toppled (end=%s): %d/%d of those got up again; the others rode it out (end=%s)' % (
        len(fell), n, sorted({c['end'] for c in fell}), len(good), len(fell), sorted({c['end'] for c in res if c['end'] not in (4, 6)})))
    if fell:
        print('       e.g. %s' % fell[0]['states'][:10])
    return bad


def cmd_render():
    import fun_render as fr
    os.makedirs(OUT, exist_ok=True)
    for nm, kw, title, sub in (('bow', dict(), '固件版 鞠躬', 'the firmware supervisor plays the bow table (MOTION state, C code in the loop)'),
                               ('bow', dict(poke_at=3.8, T_extra=12.0), '固件版 鞠躬 + 被戳', 'poked during the bow: MOTION -> FALLEN -> learned get-up -> STAND')):
        c = run_motion(nm, 1.0, seed=7, randomize=False, **kw)
        clip = dict(name='fw_bow' if not kw else 'fw_bow_poke', title=title, subtitle=sub, env=c['r'].env, frames=c['r'].frames,
                    view=dict(azim=150, elev=-12, dist=0.5, lookat_z=0.075), T=c['r'].frames[-1][0])
        print(fr.render_clip(clip, OUT, gif=False), c['states'][:10])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'abort', 'render'])
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    if a.cmd == 'check':
        sys.exit(1 if cmd_check(a.n) else 0)
    elif a.cmd == 'abort':
        sys.exit(1 if cmd_abort(a.n) else 0)
    else:
        cmd_render()
