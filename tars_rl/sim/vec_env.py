"""Multi-process vectorised TarsEnv (several envs per worker to amortise IPC)."""
import multiprocessing as mp
import numpy as np

from tars_env import TarsEnv, OBS_DIM, PRIV_DIM, ACT_DIM


def _worker(conn, env_kwargs, n_envs, seed0, env_cls=None):
    cls = env_cls or TarsEnv                        # env_cls: a TarsEnv subclass (e.g. GetupEnv); the default is the original environment
    envs = [cls(seed=seed0 + i, **env_kwargs) for i in range(n_envs)]
    ep_ret = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=np.int32)
    obs = np.zeros((n_envs, OBS_DIM), np.float32)
    priv = np.zeros((n_envs, PRIV_DIM), np.float32)
    for i, e in enumerate(envs):
        obs[i], priv[i] = e.reset()
    conn.send((obs.copy(), priv.copy()))
    while True:
        cmd, data = conn.recv()
        if cmd == 'step':
            acts = data
            rew = np.zeros(n_envs, np.float32)
            done = np.zeros(n_envs, np.bool_)
            trunc = np.zeros(n_envs, np.bool_)
            fobs = np.zeros((n_envs, OBS_DIM), np.float32)
            fpriv = np.zeros((n_envs, PRIV_DIM), np.float32)
            stats = []
            for i, e in enumerate(envs):
                o, r, d, info = e.step(acts[i])
                rew[i] = r
                ep_ret[i] += r
                ep_len[i] += 1
                tr = info['truncated']
                if d or tr:
                    done[i] = d
                    trunc[i] = tr
                    fobs[i] = o
                    fpriv[i] = info['priv']
                    stats.append(dict(ret=float(ep_ret[i]), len=int(ep_len[i]), fell=bool(d and not info.get('success', False)),
                                      success=bool(info.get('success', False)), kind=e.pose_kind,
                                      tilt=float(e.max_tilt_seen), tsucc=float(info.get('tsucc', -1.0)),
                                      up_end=float(e.d.xmat[e.id_torso].reshape(3, 3)[2, 2]), plant=getattr(e, 'plant_name', '')))
                    ep_ret[i] = 0
                    ep_len[i] = 0
                    o, p = e.reset()
                    obs[i], priv[i] = o, p
                else:
                    obs[i] = o
                    priv[i] = info['priv']
            conn.send((obs.copy(), priv.copy(), rew, done, trunc, fobs, fpriv, stats))
        elif cmd == 'cfg':
            for e in envs:
                e.cfg.update(data)
            conn.send(True)
        elif cmd == 'close':
            conn.close()
            return


class VecTars:
    def __init__(self, n_workers, envs_per_worker, env_kwargs, seed=0, env_cls=None):
        ctx = mp.get_context('spawn')
        self.n_workers, self.k = n_workers, envs_per_worker
        self.n = n_workers * envs_per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, env_kwargs, envs_per_worker, seed + 1000 * w, env_cls), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        res = [c.recv() for c in self.conns]
        self.obs = np.concatenate([r[0] for r in res])
        self.priv = np.concatenate([r[1] for r in res])

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        for w, c in enumerate(self.conns):
            c.send(('step', actions[w * self.k:(w + 1) * self.k]))
        res = [c.recv() for c in self.conns]
        self.obs = np.concatenate([r[0] for r in res])
        self.priv = np.concatenate([r[1] for r in res])
        rew = np.concatenate([r[2] for r in res])
        done = np.concatenate([r[3] for r in res])
        trunc = np.concatenate([r[4] for r in res])
        fobs = np.concatenate([r[5] for r in res])
        fpriv = np.concatenate([r[6] for r in res])
        stats = [s for r in res for s in r[7]]
        return self.obs, self.priv, rew, done, trunc, fobs, fpriv, stats

    def set_cfg(self, **kw):
        for c in self.conns:
            c.send(('cfg', kw))
        for c in self.conns:
            c.recv()

    def close(self):
        for c in self.conns:
            try:
                c.send(('close', None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
