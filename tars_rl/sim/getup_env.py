"""Get-up task for reinforcement learning (a subclass of TarsEnv: the original environment is untouched).

What it adds to TarsEnv(task='getup'):
  * the policy acts every `action_repeat` physics-control steps (5 = 10 Hz): coherent exploration, short horizon; the action is held in between
  * a servo speed limit on the commanded targets (hips `rate_hip` rad/s, lift `rate_lift` m/s): what the real servo speed register / the firmware supervisor impose
  * the elapsed time as an observation (obs[0] = t / 10 s; the cmd slot is unused in the get-up task): a feed-forward policy can sequence a multi-stage motion
  * a mix of plants per episode: the original simulator and the real-calibrated one (servo fit, command delay, IMU lag), `plant_mix` = probability of the calibrated one
  * the episode does NOT end at the first success (positive per-step rewards would otherwise punish finishing): `info['success']` / `info['tsucc']` record the first stable stand
  * the previous-action state starts at the actual servo positions (no phantom jump at the first step)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tars_env import TarsEnv, DT, HIP_H, DEFAULT_CFG        # noqa: E402

HIP = 2.4          # rad, action 1.0 -> 2.4 rad hip target (the get-up keyframes use the same scale)
LIFT = 0.014       # m

GETUP_DEFAULTS = dict(
    task='getup', hip_scale=HIP, lift_scale=LIFT, max_steps=600,        # 12 s
    randomize=True, difficulty=1.0, push_prob_per_s=0.0, getup_strict=True,
    action_repeat=5, time_obs=True, rate_hip=8.0, rate_lift=0.25,
    plant_mix=0.5, plant_cal=None, continue_after_success=True,
    reward_mode='v1', time_cost=0.01, success_bonus=30.0, act_pen=0.0,
    imu_clip=False, vel_cap=0.0, vel_pen=0.0, p_weak=0.0)

_PLANT_KEYS = ('servo_cal', 'cmd_delay_ms', 'cmd_delay_jitter_ms', 'imu_lag_samples', 'imu_lag_jitter', 'tau_range')


class GetupEnv(TarsEnv):
    def __init__(self, seed=0, **kw):
        cfg = dict(GETUP_DEFAULTS)
        cfg.update(kw)
        self.plant_name = 'orig'
        self.succeeded = False
        self.t_success = None
        super().__init__(seed=seed, **cfg)

    # ------------------------------------------------------------------ plant mix, start state
    def _pick_plant(self):
        cal = self.cfg.get('plant_cal')
        mix = self.cfg['plant_mix']
        use = cal is not None and (mix >= 1.0 or (mix > 0.0 and self.rng.random() < mix))        # no random draw for 0 / 1: the same seed gives the same plant as a plain TarsEnv
        for k in _PLANT_KEYS:
            self.cfg[k] = DEFAULT_CFG[k]
        if use:
            for k in _PLANT_KEYS:
                if k in cal:
                    self.cfg[k] = cal[k]
        self.plant_name = 'cal' if use else 'orig'
        if use and self.cfg['p_weak'] > 0.0 and self.rng.random() < self.cfg['p_weak']:
            self.cfg['tau_range'] = (0.80, 1.0)                  # a low battery / weak servo now and then

    def _init_state(self):
        self._pick_plant()
        self.succeeded = False
        self.t_success = None
        super()._init_state()
        d = self.d
        a0 = np.array([d.qpos[self.qadr['hip_l']] / HIP, d.qpos[self.qadr['hip_r']] / HIP, d.qpos[self.qadr['lift']] / LIFT])
        self.a_prev = np.clip(a0, -1.0, 1.0)
        self.a_pol_prev = self.a_prev.copy()
        self.rate_vec = np.array([self.cfg['rate_hip'] * DT / HIP, self.cfg['rate_hip'] * DT / HIP, self.cfg['rate_lift'] * DT / LIFT])
        self.prev_frame = self._frame()
        self._phi_prev = self._phi(d.xmat[self.id_torso].reshape(3, 3)[2, 2])

    # ------------------------------------------------------------------ sensors: the real QMI8658 saturates (accelerometer +-2 g, gyro +-1000 dps)
    def _read_imu(self):
        acc, gyr = super()._read_imu()
        if self.cfg['imu_clip']:
            acc = np.clip(acc, -19.62, 19.62)
            gyr = np.clip(gyr, -17.45, 17.45)
        return acc, gyr

    # ------------------------------------------------------------------ observation
    def _obs(self):
        obs = super()._obs()
        if self.cfg['time_obs']:
            obs[0] = min(self.t / 10.0, 1.5)
        return obs

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        tot = 0.0
        for _ in range(int(self.cfg['action_repeat'])):
            ai = np.clip(a, self.a_prev - self.rate_vec, self.a_prev + self.rate_vec)
            obs, r, done, info = super().step(ai)
            tot += r
            if done or info['truncated']:
                break
        if self.cfg['act_pen'] > 0.0:                    # raw actions beyond the clip range earn nothing and kill the gradient: keep the policy mean inside [-1, 1]
            tot -= self.cfg['act_pen'] * float(np.sum(np.maximum(np.abs(np.asarray(action, dtype=np.float64)) - 1.0, 0.0) ** 2))
        return obs, tot, done, info

    # ------------------------------------------------------------------ reward (same terms as TarsEnv's get-up reward; no early termination at the first success)
    def _phi(self, up_dot):
        """potential of the state for the 'v2' reward: 10 upright + 5 height + 8 (upright and arms at the nominal pose)"""
        d = self.d
        u = max(0.0, min(1.0, up_dot))
        h = min(d.qpos[2] / HIP_H, 1.0)
        n = float(np.exp(-(abs(d.qpos[8]) + abs(d.qpos[9])) / 0.8 - abs(d.qpos[7]) / 0.01))
        return 10.0 * u + 5.0 * h + 8.0 * u * u * n

    def _reward_v2(self, a, R, up_dot):
        """cost-to-go formulation: potential-based progress, a time cost, a success bonus, and the episode ends at the first stable stand"""
        d, cfg = self.d, self.cfg
        qf = d.actuator_force[:3] / self.tau_nom
        w = d.qvel[3:6]
        da = a - self.a_pol_prev
        h = min(d.qpos[2] / HIP_H, 1.0)
        nominal = (abs(d.qpos[8]) < 0.35) and (abs(d.qpos[9]) < 0.35) and (abs(d.qpos[7]) < 0.012)
        stable = (up_dot > 0.97) and (d.qpos[2] > 0.066) and (np.linalg.norm(w) < 0.6) and nominal
        self.stand_time = self.stand_time + self.dt if stable else 0.0
        phi = self._phi(up_dot)
        r = (phi - self._phi_prev) - cfg['time_cost'] - 0.002 * float(w @ w) - cfg['w_rate'] * float(da @ da) - 0.01 * float(qf @ qf)
        if cfg['vel_pen'] > 0.0:                          # keep the arms within the speed the servos are known to reach (the fit is verified up to ~7 rad/s)
            r -= cfg['vel_pen'] * float(np.sum(np.maximum(np.abs(d.qvel[7:9]) - cfg['vel_cap'], 0.0) ** 2))
        self._phi_prev = phi
        if self.stand_time > 1.0 and not self.succeeded:
            self.succeeded = True
            self.t_success = self.t
            r += cfg['success_bonus']
        info = dict(success=self.succeeded, tsucc=(self.t_success if self.succeeded else -1.0))
        return r, bool(self.succeeded and not cfg['continue_after_success']), info

    def _reward(self, a, R, up_dot):
        if self.cfg['reward_mode'] == 'v2':
            return self._reward_v2(a, R, up_dot)
        d, cfg = self.d, self.cfg
        qf = d.actuator_force[:3] / self.tau_nom
        w = d.qvel[3:6]
        da = a - self.a_pol_prev
        r_rate = -cfg['w_rate'] * float(da @ da)
        r_tau = -0.01 * float(qf @ qf)
        h = min(d.qpos[2] / HIP_H, 1.0)
        nominal = (not cfg.get('getup_strict', True)) or ((abs(d.qpos[8]) < 0.35) and (abs(d.qpos[9]) < 0.35) and (abs(d.qpos[7]) < 0.012))
        stable = (up_dot > 0.97) and (d.qpos[2] > 0.066) and (np.linalg.norm(w) < 0.6) and nominal
        self.stand_time = self.stand_time + self.dt if stable else 0.0
        r_arms = 0.0
        if up_dot > 0.9:
            r_arms = -0.6 * (d.qpos[8] ** 2 + d.qpos[9] ** 2) / 0.36 - 0.3 * (d.qpos[7] / 0.014) ** 2
        r = 1.0 * up_dot + 1.0 * h + r_arms - 0.002 * float(w @ w) + r_rate + r_tau
        if self.stand_time > 1.0 and not self.succeeded:
            self.succeeded = True
            self.t_success = self.t
            r += 20.0
        info = dict(success=self.succeeded, tsucc=(self.t_success if self.succeeded else -1.0))
        done = bool(self.succeeded and not cfg['continue_after_success'])
        return r, done, info
