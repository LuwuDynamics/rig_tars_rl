"""PPO for the get-up task (GetupEnv: 10 Hz policy, servo speed limit, elapsed time observation, original + real-calibrated plants).
Same asymmetric actor-critic as rl/ppo.py (the actor only sees what the ESP32 can measure; the critic also sees privileged simulator state).

usage:  python rl/ppo_getup.py --out runs/getup_ppo_a --iters 800
        python rl/ppo_getup.py --out runs/getup_ppo_b --init runs/getup_bc/bc.pt     (start from a behaviour-cloned actor)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('sim', 'rl', 'calib'):
    sys.path.insert(0, os.path.join(ROOT, p))
from tars_env import OBS_DIM, PRIV_DIM, ACT_DIM          # noqa: E402
from vec_env import VecTars                              # noqa: E402
from getup_env import GetupEnv                           # noqa: E402
from ppo import ActorCritic, RunningNorm                 # noqa: E402
from eval_cal import calibrated_kwargs                   # noqa: E402


def cal_plant():
    kw = calibrated_kwargs(lag=1, lag_jitter=1)
    kw.pop('ctrl_dt', None)
    return kw


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    env_kw = dict(max_steps=args.max_steps, plant_cal=cal_plant(), plant_mix=args.plant_mix, action_repeat=args.repeat,
                  rate_hip=args.rate_hip, rate_lift=args.rate_lift, time_obs=bool(args.time_obs))
    if args.env_kw:
        env_kw.update(json.loads(args.env_kw))
    vec = VecTars(args.workers, args.envs_per_worker, env_kw, seed=args.seed, env_cls=GetupEnv)
    N = vec.n
    T = args.horizon
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM, tuple(args.actor_hidden), tuple(args.critic_hidden), args.log_std0)
    onorm, pnorm = RunningNorm(OBS_DIM), RunningNorm(PRIV_DIM)
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        net.load_state_dict(ck['net'])
        onorm.mean, onorm.var, onorm.count = ck['onorm']['mean'], ck['onorm']['var'], ck['onorm']['count']
        if 'pnorm' in ck:
            pnorm.mean, pnorm.var, pnorm.count = ck['pnorm']['mean'], ck['pnorm']['var'], ck['pnorm']['count']
        if args.log_std0 is not None and args.reset_std:
            with torch.no_grad():
                net.log_std.fill_(args.log_std0)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr = args.lr
    obs, priv = vec.obs.copy(), vec.priv.copy()
    onorm.update(obs)
    pnorm.update(priv)
    log_f = open(os.path.join(args.out, 'log.csv'), 'w', newline='')
    logw = csv.writer(log_f)
    logw.writerow(['iter', 'decisions', 'time', 'mean_ret', 'succ', 'succ_orig', 'succ_cal', 'succ_back', 'succ_front', 'succ_lean', 'tsucc', 'up_end', 'lr', 'std', 'kl', 'vloss'])
    t0 = time.time()
    total = 0
    recent = []
    best = -1.0
    for it in range(1, args.iters + 1):
        B_obs = np.zeros((T, N, OBS_DIM), np.float32)
        B_priv = np.zeros((T, N, PRIV_DIM), np.float32)
        B_act = np.zeros((T, N, ACT_DIM), np.float32)
        B_logp = np.zeros((T, N), np.float32)
        B_val = np.zeros((T, N), np.float32)
        B_rew = np.zeros((T, N), np.float32)
        B_done = np.zeros((T, N), np.float32)
        R_obs = np.zeros((T, N, OBS_DIM), np.float32)
        R_priv = np.zeros((T, N, PRIV_DIM), np.float32)
        for t in range(T):
            R_obs[t], R_priv[t] = obs, priv
            on = onorm(obs).astype(np.float32)
            pn = pnorm(priv).astype(np.float32)
            with torch.no_grad():
                ot, pt = torch.from_numpy(on), torch.from_numpy(pn)
                dist = net.dist(ot)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = net.value(ot, pt)
            B_obs[t], B_priv[t], B_act[t] = on, pn, a.numpy()
            B_logp[t], B_val[t] = logp.numpy(), v.numpy()
            obs, priv, rew, done, trunc, fobs, fpriv, stats = vec.step(a.numpy())
            if trunc.any():
                idx = np.where(trunc)[0]
                with torch.no_grad():
                    fv = net.value(torch.from_numpy(onorm(fobs[idx]).astype(np.float32)),
                                   torch.from_numpy(pnorm(fpriv[idx]).astype(np.float32))).numpy()
                rew = rew.copy()
                rew[idx] += args.gamma * fv
            B_rew[t] = rew
            B_done[t] = (done | trunc).astype(np.float32)
            recent.extend(stats)
            total += N
        if it <= args.freeze_norm * args.iters:
            onorm.update(R_obs.reshape(-1, OBS_DIM))
            pnorm.update(R_priv.reshape(-1, PRIV_DIM))
        with torch.no_grad():
            last_v = net.value(torch.from_numpy(onorm(obs).astype(np.float32)), torch.from_numpy(pnorm(priv).astype(np.float32))).numpy()
        adv = np.zeros((T, N), np.float32)
        lastgae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B_val[t + 1]
            nonterm = 1.0 - B_done[t]
            delta = B_rew[t] + args.gamma * nv * nonterm - B_val[t]
            lastgae = delta + args.gamma * args.lam * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + B_val
        f_obs = torch.from_numpy(B_obs.reshape(-1, OBS_DIM))
        f_priv = torch.from_numpy(B_priv.reshape(-1, PRIV_DIM))
        f_act = torch.from_numpy(B_act.reshape(-1, ACT_DIM))
        f_logp = torch.from_numpy(B_logp.reshape(-1))
        f_adv = torch.from_numpy(adv.reshape(-1))
        f_ret = torch.from_numpy(ret.reshape(-1))
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        M = f_obs.shape[0]
        kls, vls = [], []
        for ep in range(args.epochs):
            perm = torch.randperm(M)
            for mb in perm.chunk(args.minibatches):
                d = net.dist(f_obs[mb])
                nlogp = d.log_prob(f_act[mb]).sum(-1)
                ratio = (nlogp - f_logp[mb]).exp()
                s1 = ratio * f_adv[mb]
                s2 = ratio.clamp(1 - args.clip, 1 + args.clip) * f_adv[mb]
                pl = -torch.min(s1, s2).mean()
                v = net.value(f_obs[mb], f_priv[mb])
                vl = ((v - f_ret[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kls.append((f_logp[mb] - nlogp).mean().item())
                    vls.append(vl.item())
            kl = float(np.mean(kls))
            if args.adaptive_lr:
                if kl > 2.0 * args.kl_target:
                    lr = max(1e-5, lr / 1.5)
                elif kl < 0.5 * args.kl_target:
                    lr = min(args.lr_max, lr * 1.5)
                for g in opt.param_groups:
                    g['lr'] = lr
        rs = recent[-600:]

        def rate(sel):
            xs = [s['success'] for s in rs if sel(s)]
            return float(np.mean(xs)) if xs else float('nan')
        succ = rate(lambda s: True)
        mean_ret = float(np.mean([s['ret'] for s in rs])) if rs else float('nan')
        ts = [s['tsucc'] for s in rs if s['success'] and s['tsucc'] >= 0]
        std = float(net.log_std.exp().mean().item())
        row = [it, total, round(time.time() - t0, 1), mean_ret, succ, rate(lambda s: s['plant'] == 'orig'), rate(lambda s: s['plant'] == 'cal'),
               rate(lambda s: s['kind'] == 'back'), rate(lambda s: s['kind'] == 'front'), rate(lambda s: s['kind'].startswith('lean')),
               float(np.mean(ts)) if ts else float('nan'), float(np.mean([s['up_end'] > 0.9 for s in rs])) if rs else float('nan'), lr, std, kl, float(np.mean(vls))]
        logw.writerow(row)
        log_f.flush()
        if it % args.print_every == 0 or it == 1:
            print(f'it {it:4d} dec {total/1e6:6.2f}M t {time.time()-t0:6.0f}s ret {mean_ret:8.1f} succ {succ:4.2f} (orig {row[5]:4.2f} cal {row[6]:4.2f} | back {row[7]:4.2f} front {row[8]:4.2f} lean {row[9]:4.2f}) '
                  f'tsucc {row[10]:5.2f}s up_end {row[11]:4.2f} std {std:.2f} lr {lr:.1e} kl {kl:.4f}', flush=True)
        if it % args.save_every == 0 or it == args.iters:
            ck = dict(net=net.state_dict(), onorm=dict(mean=onorm.mean, var=onorm.var, count=onorm.count),
                      pnorm=dict(mean=pnorm.mean, var=pnorm.var, count=pnorm.count), args=vars(args), it=it)
            torch.save(ck, os.path.join(args.out, 'last.pt'))
            score = succ if not np.isnan(succ) else -1.0
            if score >= best and it >= 20:
                best = score
                torch.save(ck, os.path.join(args.out, 'best.pt'))
    vec.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/getup_ppo')
    ap.add_argument('--iters', type=int, default=800)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--envs_per_worker', type=int, default=32)
    ap.add_argument('--horizon', type=int, default=48)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--minibatches', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--adaptive_lr', type=int, default=1)
    ap.add_argument('--lr_max', type=float, default=5e-4)
    ap.add_argument('--kl_target', type=float, default=0.012)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--log_std0', type=float, default=-0.6)
    ap.add_argument('--reset_std', type=int, default=0, help='with --init: reset the exploration std to --log_std0')
    ap.add_argument('--actor_hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--critic_hidden', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--max_steps', type=int, default=600, help='base 20 ms steps per episode (600 = 12 s)')
    ap.add_argument('--repeat', type=int, default=5, help='action repeat (5 = the policy acts at 10 Hz)')
    ap.add_argument('--rate_hip', type=float, default=8.0)
    ap.add_argument('--rate_lift', type=float, default=0.25)
    ap.add_argument('--time_obs', type=int, default=1)
    ap.add_argument('--plant_mix', type=float, default=0.5, help='probability of the real-calibrated plant per episode')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_every', type=int, default=20)
    ap.add_argument('--print_every', type=int, default=5)
    ap.add_argument('--freeze_norm', type=float, default=0.5)
    ap.add_argument('--env_kw', default='')
    ap.add_argument('--init', default='')
    train(ap.parse_args())
