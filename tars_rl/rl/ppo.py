"""Compact PPO (asymmetric actor-critic) for TarsEnv.  The actor only sees what the ESP32 can measure; the
critic additionally sees privileged simulator state.  Observation normalisation is folded into the first
actor layer on export so the firmware needs no running statistics.

usage:  python rl/ppo.py --task walk --iters 600 --out runs/walk1
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
sys.path.insert(0, os.path.join(ROOT, 'sim'))
from tars_env import OBS_DIM, PRIV_DIM, ACT_DIM          # noqa: E402
from vec_env import VecTars                              # noqa: E402


class RunningNorm:
    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(sizes, act=nn.Tanh, last_gain=0.01):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, np.sqrt(2) if i < len(sizes) - 2 else last_gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, actor_hidden=(128, 128), critic_hidden=(256, 256), log_std=-1.2):
        super().__init__()
        self.actor = mlp([obs_dim, *actor_hidden, act_dim], last_gain=0.01)
        self.critic = mlp([obs_dim + priv_dim, *critic_hidden, 1], last_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs, priv):
        return self.critic(torch.cat([obs, priv], -1)).squeeze(-1)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    env_kw = dict(task=args.task, max_steps=args.max_steps, difficulty=args.dif0)
    if args.env_kw:
        env_kw.update(json.loads(args.env_kw))
    vec = VecTars(args.workers, args.envs_per_worker, env_kw, seed=args.seed)
    N = vec.n
    T = args.horizon
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM, tuple(args.actor_hidden), tuple(args.critic_hidden), args.log_std0)
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        net.load_state_dict(ck['net'])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    onorm, pnorm = RunningNorm(OBS_DIM), RunningNorm(PRIV_DIM)
    if args.init:
        onorm.mean, onorm.var, onorm.count = ck['onorm']['mean'], ck['onorm']['var'], ck['onorm']['count']
        pnorm.mean, pnorm.var, pnorm.count = ck['pnorm']['mean'], ck['pnorm']['var'], ck['pnorm']['count']
    lr = args.lr
    obs, priv = vec.obs.copy(), vec.priv.copy()
    onorm.update(obs)
    pnorm.update(priv)
    log_f = open(os.path.join(args.out, 'log.csv'), 'w', newline='')
    logw = csv.writer(log_f)
    logw.writerow(['iter', 'steps', 'time', 'mean_ret', 'mean_len', 'fall_rate', 'success_rate', 'lr', 'std', 'dif', 'kl', 'vloss'])
    t0 = time.time()
    total_steps = 0
    recent = []
    best = -1e9
    for it in range(1, args.iters + 1):
        dif = min(1.0, args.dif0 + (1.0 - args.dif0) * it / max(1, args.dif_iters))
        if it == 1 or it % 5 == 0:
            vec.set_cfg(difficulty=dif)
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
            if trunc.any():  # bootstrap time-limit truncations with V(final state)
                idx = np.where(trunc)[0]
                with torch.no_grad():
                    fv = net.value(torch.from_numpy(onorm(fobs[idx]).astype(np.float32)),
                                   torch.from_numpy(pnorm(fpriv[idx]).astype(np.float32))).numpy()
                rew = rew.copy()
                rew[idx] += args.gamma * fv
            B_rew[t] = rew
            B_done[t] = (done | trunc).astype(np.float32)
            recent.extend(stats)
            total_steps += N
        if it <= args.freeze_norm * args.iters:
            onorm.update(R_obs.reshape(-1, OBS_DIM))
            pnorm.update(R_priv.reshape(-1, PRIV_DIM))
        with torch.no_grad():
            last_v = net.value(torch.from_numpy(onorm(obs).astype(np.float32)),
                               torch.from_numpy(pnorm(priv).astype(np.float32))).numpy()
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
        rs = recent[-400:]
        mean_ret = float(np.mean([s['ret'] for s in rs])) if rs else float('nan')
        mean_len = float(np.mean([s['len'] for s in rs])) if rs else float('nan')
        fall = float(np.mean([s['fell'] for s in rs])) if rs else float('nan')
        succ = float(np.mean([s['success'] for s in rs])) if rs else float('nan')
        std = float(net.log_std.exp().mean().item())
        logw.writerow([it, total_steps, round(time.time() - t0, 1), mean_ret, mean_len, fall, succ, lr, std, dif, kl, float(np.mean(vls))])
        log_f.flush()
        if it % args.print_every == 0 or it == 1:
            print(f'it {it:4d} steps {total_steps/1e6:6.2f}M t {time.time()-t0:6.0f}s ret {mean_ret:8.2f} len {mean_len:6.1f} '
                  f'fall {fall:4.2f} succ {succ:4.2f} lr {lr:.1e} std {std:.2f} dif {dif:.2f} kl {kl:.4f}', flush=True)
        if it % args.save_every == 0 or it == args.iters:
            ck = dict(net=net.state_dict(), onorm=dict(mean=onorm.mean, var=onorm.var, count=onorm.count),
                      pnorm=dict(mean=pnorm.mean, var=pnorm.var, count=pnorm.count), args=vars(args), it=it)
            torch.save(ck, os.path.join(args.out, 'last.pt'))
            if mean_ret > best and dif >= 0.999:
                best = mean_ret
                torch.save(ck, os.path.join(args.out, 'best.pt'))
    vec.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='walk')
    ap.add_argument('--out', default='runs/walk')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--envs_per_worker', type=int, default=32)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--minibatches', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--adaptive_lr', type=int, default=1)
    ap.add_argument('--lr_max', type=float, default=5e-4)
    ap.add_argument('--kl_target', type=float, default=0.012)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.002)
    ap.add_argument('--log_std0', type=float, default=-1.2)
    ap.add_argument('--actor_hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--critic_hidden', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--dif0', type=float, default=0.2)
    ap.add_argument('--dif_iters', type=int, default=150)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_every', type=int, default=25)
    ap.add_argument('--print_every', type=int, default=10)
    ap.add_argument('--freeze_norm', type=float, default=0.7, help='stop updating obs statistics after this fraction of iters')
    ap.add_argument('--env_kw', default='')
    ap.add_argument('--init', default='')
    train(ap.parse_args())
