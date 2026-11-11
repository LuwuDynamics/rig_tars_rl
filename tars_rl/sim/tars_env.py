"""Gym-style MuJoCo environment of the Tars robot, built so that everything the policy sees can also be
produced by the ESP32-S3 firmware (IMU accel+gyro -> Mahony filter, servo position read-back, previous action).

tasks
    'walk'  : stand / walk at commanded (vx, wz) while being pushed.  Obs includes command + gait clock.
    'getup' : start lying on back / front / leaning, get to a stable upright stance.

time base:  physics 2 ms (same as the firmware robot_task), policy 20 ms (50 Hz), IMU filter 100 Hz.
"""
import copy
import math
import os
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, 'model', 'tars.xml')

COUNT_RAD = math.radians(300.0 / 1024.0)     # one servo count
DT_PHYS = 0.002
DECIM = 10
DT = DT_PHYS * DECIM
G = 9.81
HIP_H = 0.0728                                # hip-axis height when standing (torso origin)

OBS_FRAME = 12                                # gvec3 gyro3 q3 a_prev3
OBS_DIM = 2 + 2 + 2 * OBS_FRAME               # cmd2 clock2 + two frames
ACT_DIM = 3
PRIV_DIM = 18


_MODEL_CACHE = {}


def load_model(path):
    """compile once per process, hand out private copies (domain randomisation edits the arrays in place)"""
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = mujoco.MjModel.from_xml_path(path)
    return copy.deepcopy(_MODEL_CACHE[path])


# ----------------------------------------------------------------------------------------------- IMU filter
class Mahony:
    """Mahony complementary filter on (gyro [rad/s], accel [m/s^2]) with accel gating.
    Identical algorithm is implemented in C for the firmware (tars_rl_runtime.c)."""

    def __init__(self, kp=2.0, sig_a=2.0):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.kp, self.sig_a = kp, sig_a

    def init_from_acc(self, acc):
        n = np.linalg.norm(acc)
        u = acc / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        th = -math.asin(max(-1.0, min(1.0, u[0])))
        ph = math.atan2(u[1], u[2])
        cy, sy = 1.0, 0.0
        cp, sp = math.cos(th / 2), math.sin(th / 2)
        cr, sr = math.cos(ph / 2), math.sin(ph / 2)
        self.q = np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                           cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

    def update(self, gyro, acc, dt):
        qw, qx, qy, qz = self.q
        vx = 2.0 * (qx * qz - qw * qy)
        vy = 2.0 * (qw * qx + qy * qz)
        vz = qw * qw - qx * qx - qy * qy + qz * qz
        gx, gy, gz = gyro
        n = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
        if n > 1e-3:
            ax, ay, az = acc[0] / n, acc[1] / n, acc[2] / n
            ex = ay * vz - az * vy
            ey = az * vx - ax * vz
            ez = ax * vy - ay * vx
            w = math.exp(-0.5 * ((n - G) / self.sig_a) ** 2)
            k = self.kp * w
            gx += k * ex
            gy += k * ey
            gz += k * ez
        h = 0.5 * dt
        nqw = qw + (-qx * gx - qy * gy - qz * gz) * h
        nqx = qx + (qw * gx + qy * gz - qz * gy) * h
        nqy = qy + (qw * gy - qx * gz + qz * gx) * h
        nqz = qz + (qw * gz + qx * gy - qy * gx) * h
        inv = 1.0 / math.sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
        self.q = np.array([nqw * inv, nqx * inv, nqy * inv, nqz * inv])

    def up_body(self):
        qw, qx, qy, qz = self.q
        return np.array([2.0 * (qx * qz - qw * qy), 2.0 * (qw * qx + qy * qz),
                         qw * qw - qx * qx - qy * qy + qz * qz])


# ----------------------------------------------------------------------------------------------- env
DEFAULT_CFG = dict(
    task='walk',
    xml=None,                 # alternative model file (design variants)
    tau_nominal=1.0,          # used only when randomize=False
    tau_range=(1.0, 1.4),     # supply 2S Li-ion 6.6-8.4 V (battery level code in tars_board.cc), minus sag; 1.0 = 6 V datasheet point
    max_steps=500,
    hip_scale=0.6,            # rad, action 1.0 -> 0.6 rad hip target
    lift_scale=0.014,         # m
    gait_freq=1.25,
    vx_range=(-0.04, 0.08),   # m/s command range
    wz_range=(-0.8, 0.8),
    p_stand=0.3,
    push_prob_per_s=0.4,      # pushes per second (mean)
    push_impulse=(0.0, 0.12),  # N s, horizontal impulse on the torso
    push_fixed=None,          # evaluator: (angle_rad, impulse_Ns) for every push
    cmd_fixed=None,           # evaluator: (vx, wz) held for the whole episode
    no_terminate=False,       # walk task never terminates (used for supervisor tests)
    getup_strict=True,        # success requires arms back at the nominal stance
    getup_init=None,          # dict(kind='back'|'front'|'lean_f'|'lean_b', yaw, hip=(l,r), lift) for deterministic starts
    push_steps=2,             # push lasts this many 20 ms policy steps
    randomize=True,
    difficulty=1.0,           # curriculum scalar 0..1 for pushes / randomisation
    fall_up=0.5,              # terminate (walk) when cos(tilt) below this
    action_mode='direct',     # 'direct' | 'residual' (on the firmware gait) | 'es_residual' (on a CMA-ES feedback gait)
    es_params=None,           # dict of the 15 CMA-ES gait parameters (es_residual)
    es_ramp=1.0,
    obs_q_mode='meas',        # 'meas' (servo read-back) | 'cmd' (previous commanded targets; no read-back needed)
    res_hip=0.25,             # rad of hip residual authority at |a|=1
    res_lift=0.008,           # m of lift residual authority at |a|=1
    amp_per_vx=1000.0,        # firmware-gait amplitude [counts] per m/s of forward command
    amp_per_wz=30.0,          # counts per rad/s of yaw command
    yaw_sign=-1.0,
    # ---- real-robot calibration (2026-10-28 data; all off by default = the original simulator) ----
    servo_cal=None,           # dict(kp=, kd=, arm=, lift_kp=, lift_kd=, tau=): multipliers on the nominal servo model (fitted to the real read-back)
    cmd_delay_ms=None,        # pure command->servo delay [ms] (replaces the 0..8 ms random latency)
    cmd_delay_jitter_ms=0.0,  # +- uniform jitter on that delay under randomisation
    imu_lag_samples=0,        # the controller sees the attitude / gyro that many 10 ms IMU samples old (read + filter + scheduling latency)
    imu_lag_jitter=0,         # +- samples under randomisation
    ctrl_dt=None,             # policy / controller period [s]: None = 0.02 (50 Hz, the original), 0.01 = 100 Hz (the firmware option `learn tune ... 100`)
    com_shift=None,           # (dx, dy, dz) [m] added to the torso centre of mass on top of the randomisation (sensitivity tests)
    act_slew=None,            # max change of the (normalised) applied action per policy step (None = unlimited): the policy cannot chatter at the Nyquist rate
    w_rate=0.02,              # weight of the action-rate penalty in the walk reward
    w_rate_calm=None,         # if set: the rate penalty weight while the robot is calm (|w| < 0.5 rad/s, upright, > 0.6 s after the last push): quiet at rest, free to act when pushed
)


class TarsEnv:
    def __init__(self, seed=0, **kw):
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(kw)
        self.rng = np.random.default_rng(seed)
        self.m = load_model(self.cfg['xml'] or XML)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.id_torso = m.body('torso').id
        self.id_arm = [m.body('arm_l').id, m.body('arm_r').id]
        self.id_slider = m.body('slider').id
        self.g_floor = m.geom('floor').id
        self.g_boxes = [m.geom('torso_box').id, m.geom('arm_l_box').id, m.geom('arm_r_box').id]
        self.qadr = dict(lift=7, hip_l=8, hip_r=9)
        self.dadr = dict(lift=6, hip_l=7, hip_r=8)
        # nominal copies for domain randomisation
        self._nom = dict(
            body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(), body_ipos=m.body_ipos.copy(),
            geom_friction=m.geom_friction.copy(), gainprm=m.actuator_gainprm.copy(),
            biasprm=m.actuator_biasprm.copy(), forcerange=m.actuator_forcerange.copy(),
            damping=m.dof_damping.copy(), armature=m.dof_armature.copy(), frictionloss=m.dof_frictionloss.copy(),
            gravity=m.opt.gravity.copy())
        self.tau_nom = np.abs(m.actuator_forcerange[:, 1]).copy()
        self.dt = float(self.cfg['ctrl_dt'] or DT)                 # policy period; DT stays the original 50 Hz constant
        self.n_sub = int(round(self.dt / DT_PHYS))                 # physics sub-steps per policy step (10 or 5)
        assert self.n_sub in (5, 10), 'ctrl_dt must be 0.01 or 0.02'
        self.hip_scale = self.cfg['hip_scale']
        self.lift_scale = self.cfg['lift_scale']
        self.max_steps = self.cfg['max_steps']
        self.filter = Mahony()
        self.imu_hook = None
        self.steps = 0
        self._init_state()

    # ------------------------------------------------------------------ randomisation
    def _randomize(self):
        m, d, rng, nom = self.m, self.d, self.rng, self._nom
        dif = self.cfg['difficulty']
        m.body_mass[:] = nom['body_mass']
        m.body_inertia[:] = nom['body_inertia']
        m.body_ipos[:] = nom['body_ipos']
        m.geom_friction[:] = nom['geom_friction']
        m.actuator_gainprm[:] = nom['gainprm']
        m.actuator_biasprm[:] = nom['biasprm']
        m.actuator_forcerange[:] = nom['forcerange']
        m.dof_damping[:] = nom['damping']
        m.dof_armature[:] = nom['armature']
        m.dof_frictionloss[:] = nom['frictionloss']
        m.opt.gravity[:] = nom['gravity']
        pr = dict(mu=0.8, mass_scale=1.0, tau_scale=1.0, lat_substeps=0, db_counts=0.0, slope=0.0)
        if self.cfg['randomize']:
            f = dif
            mu = float(np.exp(rng.uniform(math.log(0.35), math.log(1.3))))
            for g in self.g_boxes + [self.g_floor]:
                m.geom_friction[g, 0] = mu
            pr['mu'] = mu
            # masses / inertias (robot total +-8 %, individual bodies +-10 %)
            ms = rng.uniform(0.92, 1.08)
            pr['mass_scale'] = ms
            for b in [self.id_torso, self.id_slider] + self.id_arm:
                s = ms * rng.uniform(0.9, 1.1)
                m.body_mass[b] *= s
                m.body_inertia[b] *= s
            # CoM offsets (uncertain battery / PCB position): torso +-6 mm x,z ; +-2 mm y
            m.body_ipos[self.id_torso] += rng.uniform([-0.006, -0.002, -0.006], [0.006, 0.002, 0.006]) * (0.3 + 0.7 * f)
            if self.cfg['com_shift'] is not None:
                m.body_ipos[self.id_torso] += np.asarray(self.cfg['com_shift'], float)
            # servo: torque (supply voltage 4.5..6.3 V), kp, damping, armature, friction
            ts = rng.uniform(*self.cfg['tau_range'])      # servo torque / speed scale relative to the 6 V datasheet point (~ V/6)
            pr['tau_scale'] = ts
            kps = ts * rng.uniform(0.8, 1.25)             # servo controller output is PWM duty x supply voltage -> gains scale with V too
            kds = ts * rng.uniform(0.8, 1.25)
            for i in range(3):
                kp = nom['gainprm'][i, 0] * kps
                kd = -nom['biasprm'][i, 2] * kds
                m.actuator_gainprm[i, 0] = kp
                m.actuator_biasprm[i, 1] = -kp
                m.actuator_biasprm[i, 2] = -kd
                m.actuator_forcerange[i] = nom['forcerange'][i] * ts
            arm_s = rng.uniform(0.6, 1.6)
            damp_s = rng.uniform(0.8, 1.25)   # kv = tau_stall/w_noload is the same at every voltage
            fl_s = rng.uniform(0.5, 2.5)
            for dof in (6, 7, 8):
                m.dof_armature[dof] *= arm_s
                m.dof_damping[dof] *= damp_s
                m.dof_frictionloss[dof] *= fl_s
            pr['lat_substeps'] = int(rng.integers(0, 5))   # 0..8 ms command latency
            if self.cfg['servo_cal']:
                self._apply_servo_cal(self.cfg['servo_cal'])
            pr['db_counts'] = float(rng.uniform(0.0, 2.0))  # servo dead-band / backlash [counts]
            pr['slope'] = float(rng.uniform(-3, 3) * f)     # deg fore/aft slope
            sl = math.radians(pr['slope'])
            m.opt.gravity[:] = [G * math.sin(sl), 0.0, -G * math.cos(sl)]
        elif self.cfg.get('tau_nominal', 1.0) != 1.0:     # deterministic runs at a given supply voltage (1.2 ~ 7.2 V)
            tn = self.cfg['tau_nominal']
            for i in range(3):
                m.actuator_gainprm[i, 0] = nom['gainprm'][i, 0] * tn
                m.actuator_biasprm[i, 1] = nom['biasprm'][i, 1] * tn
                m.actuator_biasprm[i, 2] = nom['biasprm'][i, 2] * tn
                m.actuator_forcerange[i] = nom['forcerange'][i] * tn
            pr['tau_scale'] = tn
        if not self.cfg['randomize'] and self.cfg['servo_cal']:
            self._apply_servo_cal(self.cfg['servo_cal'])
        # command delay / IMU lag (real-robot calibration)
        pr['delay_substeps'] = None
        if self.cfg['cmd_delay_ms'] is not None:
            jit = self.cfg['cmd_delay_jitter_ms'] if self.cfg['randomize'] else 0.0
            pr['delay_substeps'] = max(0, int(round((self.cfg['cmd_delay_ms'] + rng.uniform(-jit, jit)) / (DT_PHYS * 1000.0))))
        lag = self.cfg['imu_lag_samples']
        if self.cfg['randomize'] and self.cfg['imu_lag_jitter']:
            lag = max(0, lag + int(rng.integers(-self.cfg['imu_lag_jitter'], self.cfg['imu_lag_jitter'] + 1)))
        pr['imu_lag'] = int(lag)
        mujoco.mj_setConst(m, d)
        self.pr = pr
        # sensor error model
        e = self.cfg['randomize']
        self.acc_bias = rng.normal(0, 0.15, 3) if e else np.zeros(3)
        self.gyro_bias = rng.normal(0, 0.012, 3) if e else np.zeros(3)
        self.acc_noise = 0.12 if e else 0.0
        self.gyro_noise = 0.015 if e else 0.0
        dpsi = rng.normal(0, math.radians(1.5), 3) if e else np.zeros(3)
        self.R_mount = np.eye(3) + np.array([[0, -dpsi[2], dpsi[1]], [dpsi[2], 0, -dpsi[0]], [-dpsi[1], dpsi[0], 0]])
        self.q_noise = 0.6 if e else 0.0      # counts rms on read-back
        self.filter.kp = float(rng.uniform(1.0, 3.5)) if e else 2.0

    def _apply_servo_cal(self, cal):
        m = self.m
        for i in range(3):
            kp_s = cal.get('kp' if i < 2 else 'lift_kp', 1.0)
            kd_s = cal.get('kd' if i < 2 else 'lift_kd', 1.0)
            m.actuator_gainprm[i, 0] *= kp_s
            m.actuator_biasprm[i, 1] *= kp_s
            m.actuator_biasprm[i, 2] *= kd_s
            m.actuator_forcerange[i] *= cal.get('tau', 1.0)
        for dof in (7, 8):
            m.dof_armature[dof] *= cal.get('arm', 1.0)
        if 'fric' in cal:
            for dof in (7, 8):
                m.dof_frictionloss[dof] *= cal['fric']

    # ------------------------------------------------------------------ state init
    def _init_state(self):
        self._randomize()
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetData(m, d)
        task = self.cfg['task']
        d.qpos[2] = HIP_H + 0.0005
        d.qpos[3:7] = [1, 0, 0, 0]
        self.pose_kind = 'stand'
        if task == 'walk':
            tilt = rng.normal(0, math.radians(2.0), 2)
            d.qpos[3:7] = self._quat_rpy(tilt[0], tilt[1], rng.uniform(-math.pi, math.pi))
            d.qpos[self.qadr['hip_l']] = rng.uniform(-0.1, 0.1)
            d.qpos[self.qadr['hip_r']] = rng.uniform(-0.1, 0.1)
            d.qpos[self.qadr['lift']] = rng.uniform(-0.006, 0.006)
        else:
            gi = self.cfg.get('getup_init')
            kinds = ['back', 'front', 'back', 'front', 'lean_f', 'lean_b']
            self.pose_kind = gi['kind'] if gi else kinds[int(rng.integers(len(kinds)))]
            yaw = gi.get('yaw', 0.0) if gi else rng.uniform(-math.pi, math.pi)
            if self.pose_kind == 'back':
                d.qpos[3:7] = self._quat_rpy(0.0, -math.pi / 2, yaw)
                d.qpos[2] = 0.045
            elif self.pose_kind == 'front':
                d.qpos[3:7] = self._quat_rpy(0.0, math.pi / 2, yaw)
                d.qpos[2] = 0.045
            else:
                ang = rng.uniform(math.radians(25), math.radians(70))
                sgn = 1.0 if self.pose_kind == 'lean_f' else -1.0
                d.qpos[3:7] = self._quat_rpy(0.0, sgn * ang, yaw)
                d.qpos[2] = 0.07
            for k in ('hip_l', 'hip_r'):
                d.qpos[self.qadr[k]] = rng.uniform(-1.2, 1.2) if rng.random() < 0.7 else 0.0
            d.qpos[self.qadr['lift']] = rng.uniform(-0.01, 0.01)
            if gi:
                d.qpos[self.qadr['hip_l']], d.qpos[self.qadr['hip_r']] = gi.get('hip', (0.0, 0.0))
                d.qpos[self.qadr['lift']] = gi.get('lift', 0.0)
        d.ctrl[0] = d.qpos[self.qadr['hip_l']]
        d.ctrl[1] = d.qpos[self.qadr['hip_r']]
        d.ctrl[2] = d.qpos[self.qadr['lift']]
        mujoco.mj_forward(m, d)
        for _ in range(150):                      # let the robot settle completely before the sensors are used
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.R_mount_init = None
        self.steps = 0
        self.a_prev = np.zeros(3)
        self.a_pol_prev = np.zeros(3)
        self.ctrl_applied = d.ctrl.copy()
        self.tgt_applied = d.ctrl.copy()
        self.phase = rng.uniform(0, 1)
        self.t = 0.0
        self.cmd = np.zeros(2)
        self._resample_cmd(force=True)
        self.push_force = np.zeros(3)
        self.push_until = -1.0
        self.last_push_t = -1e9
        self.next_push = 1.0 + rng.exponential(1.0 / max(self.cfg['push_prob_per_s'], 1e-3))
        self.stand_time = 0.0
        self.max_tilt_seen = 0.0
        # filter start-up like the real robot at boot: robot at rest, average a few accelerometer samples
        accs, gyrs = [], []
        for _ in range(5):
            mujoco.mj_step(m, d)
            a_, g_ = self._read_imu()
            accs.append(a_); gyrs.append(g_)
        self.filter.init_from_acc(np.mean(accs, axis=0))
        self.imu_gyro = gyrs[-1]
        self._imu_hist = [(self.filter.up_body().copy(), np.array(gyrs[-1]).copy())] * 64
        self._tgt_hist = [self.tgt_applied.copy()] * 32
        if self.imu_hook is not None:
            for a_, g_ in zip(accs, gyrs):
                self.imu_hook(a_, g_)
        frame = self._frame()
        self.prev_frame = frame.copy()

    @staticmethod
    def _quat_rpy(r, p, y):
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

    def _resample_cmd(self, force=False):
        if self.cfg['task'] != 'walk':
            self.cmd[:] = 0
            return
        if self.cfg.get('cmd_fixed') is not None:
            self.cmd[:] = self.cfg['cmd_fixed']
            self.cmd_until = 1e9
            return
        rng = self.rng
        if rng.random() < self.cfg['p_stand']:
            self.cmd[:] = 0
        else:
            lo, hi = self.cfg['vx_range']
            self.cmd[0] = rng.uniform(lo, hi)
            lo, hi = self.cfg['wz_range']
            self.cmd[1] = rng.uniform(lo, hi) if rng.random() < 0.5 else 0.0
        self.cmd_until = self.t + rng.uniform(3.0, 6.0)

    # ------------------------------------------------------------------ sensors
    def _read_imu(self):
        s = self.d.sensordata
        gyr = self.R_mount @ s[0:3] + self.gyro_bias + self.rng.normal(0, self.gyro_noise, 3)
        acc = self.R_mount @ s[3:6] + self.acc_bias + self.rng.normal(0, self.acc_noise, 3)
        return acc, gyr

    def _read_q(self):
        d = self.d
        q = np.array([d.qpos[8], d.qpos[9], d.qpos[7]])
        # servo read-back: integer counts (+ noise), lift expressed through crank ratio (0.293 deg/count -> ~0.13 mm)
        res = np.array([COUNT_RAD, COUNT_RAD, 0.00013])
        n = self.rng.normal(0, self.q_noise, 3) if self.cfg['randomize'] else np.zeros(3)
        return np.round(q / res + n) * res

    def _frame(self):
        up = self.up_ctrl()                     # as the controller sees it: lagged by pr['imu_lag'] IMU samples (0 = the newest = the original behaviour)
        gyr = self.gyro_ctrl()
        qn = self._read_q() if self.cfg['obs_q_mode'] == 'meas' else np.array([self.tgt_applied[0], self.tgt_applied[1], self.tgt_applied[2]])
        self.q_meas = qn
        return np.concatenate([-up, gyr * 0.25, qn[:2] / self.hip_scale, [qn[2] / self.lift_scale], self.a_prev])

    def _obs(self):
        frame = self._frame()
        moving = 1.0 if (abs(self.cmd[0]) > 1e-4 or abs(self.cmd[1]) > 1e-4) else 0.0
        clock = [math.sin(2 * math.pi * self.phase) * moving, math.cos(2 * math.pi * self.phase) * moving]
        cmd_n = [self.cmd[0] / 0.08, self.cmd[1] / 0.8]
        obs = np.concatenate([cmd_n, clock, frame, self.prev_frame]).astype(np.float32)
        self.prev_frame = frame
        return obs

    # ------------------------------------------------------------------ firmware gait as a normalised action
    def baseline_action(self):
        cfg = self.cfg
        ax = float(np.clip(cfg['amp_per_vx'] * self.cmd[0], -35.0, 35.0))
        ay = float(np.clip(cfg['amp_per_wz'] * self.cmd[1], -25.0, 25.0))
        if abs(ax) < 0.5 and abs(ay) < 0.5:
            return np.zeros(3)
        s_, c_ = math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)
        if abs(ay) >= abs(ax):
            ratio = abs(ay) / 25.0
            ql = cfg['yaw_sign'] * ay * s_
            qr = -cfg['yaw_sign'] * ay * s_
        else:
            ratio = abs(ax) / 35.0
            ql = qr = ax * s_
        lift = (9.0 + 5.0 * ratio) * c_ / 1000.0
        return np.array([ql * COUNT_RAD / self.hip_scale, qr * COUNT_RAD / self.hip_scale, lift / self.lift_scale])

    def es_base_action(self):
        """CMA-ES feedback gait (rl/walk_cma.py) as a normalised action; ramp-in over es_ramp seconds, same limits as the search"""
        P = self.cfg['es_params']
        if abs(self.cmd[0]) < 1e-4 and abs(self.cmd[1]) < 1e-4:
            return np.zeros(3)
        t = self.t
        ph = 2 * math.pi * P['f'] * t
        c1, s1, c2, s2 = math.cos(ph), math.sin(ph), math.cos(2 * ph), math.sin(2 * ph)
        q = P['q0'] + P['a1'] * c1 + P['b1'] * s1 + P['a2'] * c2 + P['b2'] * s2
        z = P['z0'] + P['c1'] * c1 + P['d1'] * s1 + P['c2'] * c2 + P['d2'] * s2
        up = self.filter.up_body()
        theta = -math.asin(max(-1.0, min(1.0, up[0])))
        q += P['kp_h'] * theta + P['kd_h'] * self.imu_gyro[1]
        z += P['kp_z'] * theta + P['kd_z'] * self.imu_gyro[1]
        g = min(1.0, t / self.cfg['es_ramp'])
        q = max(-0.6, min(0.6, q * g)); z = max(-0.014, min(0.014, z * g))
        return np.array([q / self.hip_scale, q / self.hip_scale, z / self.lift_scale])

    # ------------------------------------------------------------------ step
    def step(self, action):
        cfg, d, m = self.cfg, self.d, self.m
        a_pol = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if cfg['action_mode'] == 'residual':
            base = self.baseline_action()
            res = np.array([cfg['res_hip'] / self.hip_scale, cfg['res_hip'] / self.hip_scale, cfg['res_lift'] / self.lift_scale])
            a = np.clip(base + a_pol * res, -1.0, 1.0)
        elif cfg['action_mode'] == 'es_residual':
            base = self.es_base_action()
            res = np.array([cfg['res_hip'] / self.hip_scale, cfg['res_hip'] / self.hip_scale, cfg['res_lift'] / self.lift_scale])
            a = np.clip(base + a_pol * res, -1.0, 1.0)
        else:
            a = a_pol
        if cfg['act_slew']:
            a = np.clip(a, self.a_prev - cfg['act_slew'], self.a_prev + cfg['act_slew'])
        # --- command -> servo targets (position counts, dead-band hysteresis) ---
        tgt = np.array([a[0] * self.hip_scale, a[1] * self.hip_scale, a[2] * self.lift_scale])
        res = np.array([COUNT_RAD, COUNT_RAD, 0.00013])
        tq = np.round(tgt / res) * res
        db = self.pr['db_counts'] * res
        upd = np.abs(tq - self.tgt_applied) > db
        new_tgt = np.where(upd, tq, self.tgt_applied)
        self.tgt_applied = new_tgt
        lat = self.pr['lat_substeps']
        # --- pushes ---
        if self.t >= self.next_push and cfg['push_prob_per_s'] > 0:
            pf = cfg.get('push_fixed')
            if pf is not None:                      # (angle_rad, impulse_Ns) used by the evaluator
                ang, imp = pf
            else:
                imp = self.rng.uniform(*cfg['push_impulse']) * cfg['difficulty']
                ang = self.rng.uniform(0, 2 * math.pi)
                if self.rng.random() < 0.6:
                    ang = 0.0 if self.rng.random() < 0.5 else math.pi
            F = imp / (cfg['push_steps'] * DT)       # a push always lasts push_steps x 20 ms, whatever the controller period
            self.push_force = np.array([F * math.cos(ang), F * math.sin(ang), 0.0])
            self.push_until = self.t + cfg['push_steps'] * DT - 1e-6
            self.last_push_t = self.t
            self.next_push = self.t + self.rng.exponential(1.0 / cfg['push_prob_per_s'])
        # --- physics: 2 x 5 substeps, IMU sampled / filtered at 100 Hz ---
        old_ctrl = self.ctrl_applied
        # sub-steps before the new command arrives
        d.xfrc_applied[self.id_torso, :3] = self.push_force if self.t < self.push_until else 0.0
        D = self.pr.get('delay_substeps')
        ns = self.n_sub
        if D is None:
            if ns == 10:
                if lat > 0:
                    d.ctrl[:] = old_ctrl
                    mujoco.mj_step(m, d, nstep=lat)
                d.ctrl[:] = new_tgt
                mujoco.mj_step(m, d, nstep=5 - lat)
                self._imu_update()
                mujoco.mj_step(m, d, nstep=5)
                self._imu_update()
            else:                                       # 100 Hz controller: one IMU block per policy step
                if lat > 0:
                    d.ctrl[:] = old_ctrl
                    mujoco.mj_step(m, d, nstep=lat)
                d.ctrl[:] = new_tgt
                mujoco.mj_step(m, d, nstep=5 - lat)
                self._imu_update()
        else:
            # pure command delay of D physics sub-steps: sub-step s of this policy step runs with the target issued (s - D) // ns policy steps ago
            self._tgt_hist.append(new_tgt.copy())
            if len(self._tgt_hist) > 32:
                del self._tgt_hist[0]
            runs = []                                   # [relative policy step, number of sub-steps]; runs are cut every 5 sub-steps (IMU sample)
            for s_ in range(ns):
                k = (s_ - D) // ns
                if runs and runs[-1][0] == k and s_ % 5 != 0:
                    runs[-1][1] += 1
                else:
                    runs.append([k, 1])
            done = 0
            for k, n in runs:
                idx = -1 + k
                d.ctrl[:] = self._tgt_hist[idx] if -idx <= len(self._tgt_hist) else self._tgt_hist[0]
                mujoco.mj_step(m, d, nstep=n)
                done += n
                if done % 5 == 0:
                    self._imu_update()
        d.xfrc_applied[self.id_torso, :3] = 0.0
        self.ctrl_applied = new_tgt.copy()
        self.t += self.dt
        self.steps += 1
        # --- bookkeeping ---
        moving = abs(self.cmd[0]) > 1e-4 or abs(self.cmd[1]) > 1e-4
        if moving:
            self.phase = (self.phase + cfg['gait_freq'] * self.dt) % 1.0
        else:
            self.phase = 0.0
        if cfg['task'] == 'walk' and self.t >= self.cmd_until:
            self._resample_cmd()
        R = d.xmat[self.id_torso].reshape(3, 3)
        up_dot = R[2, 2]
        self.max_tilt_seen = max(self.max_tilt_seen, math.degrees(math.acos(max(-1, min(1, up_dot)))))
        reward, done, info = self._reward(a_pol, R, up_dot)
        self.a_prev = a.copy()
        self.a_pol_prev = a_pol.copy()
        obs = self._obs()
        trunc = (self.steps >= self.max_steps) and not done
        info['truncated'] = trunc
        info['priv'] = self._priv(R)
        return obs, reward, done, info

    def up_ctrl(self):
        """attitude ('up' in the body frame) as the controller sees it: lagged by pr['imu_lag'] IMU samples (default 0 = the newest)"""
        return self._imu_hist[-1 - min(self.pr.get('imu_lag', 0), len(self._imu_hist) - 1)][0]

    def gyro_ctrl(self):
        return self._imu_hist[-1 - min(self.pr.get('imu_lag', 0), len(self._imu_hist) - 1)][1]

    def _imu_update(self):
        acc, gyr = self._read_imu()
        self.filter.update(gyr, acc, 0.01)
        self.imu_gyro = gyr
        self._imu_hist.append((self.filter.up_body().copy(), np.array(gyr).copy()))
        if len(self._imu_hist) > 64:
            del self._imu_hist[0]
        if self.imu_hook is not None:             # lets an external (C firmware) filter see the raw samples
            self.imu_hook(acc, gyr)

    # ------------------------------------------------------------------ privileged info for the critic
    def _priv(self, R):
        d = self.d
        v_b = R.T @ d.qvel[0:3]
        ncon = d.ncon
        c = np.zeros(3)
        if ncon:
            g1 = d.contact.geom1[:ncon]
            g2 = d.contact.geom2[:ncon]
            for k, gb in enumerate(self.g_boxes):
                c[k] = float(np.any(((g1 == gb) & (g2 == self.g_floor)) | ((g2 == gb) & (g1 == self.g_floor))))
        pf = self.push_force[:2] if self.t < self.push_until else np.zeros(2)
        p = self.pr
        z = np.concatenate([v_b, R[2, :], c, [p['mu'], p['mass_scale'], p['tau_scale'], p['lat_substeps'] / 4.0,
                                              p['db_counts'] / 2.0, p['slope'] / 3.0], pf * 0.2,
                            [d.qpos[2] / HIP_H]])
        return z.astype(np.float32)

    # ------------------------------------------------------------------ rewards
    def _reward(self, a, R, up_dot):
        d, cfg = self.d, self.cfg
        task = cfg['task']
        qf = d.actuator_force[:3] / self.tau_nom
        w = d.qvel[3:6]
        da = a - self.a_pol_prev
        wr = cfg['w_rate']
        if cfg['w_rate_calm'] is not None and task == 'walk':
            if float(np.linalg.norm(w)) < 0.5 and up_dot > 0.97 and self.t - self.last_push_t > 0.6:
                wr = cfg['w_rate_calm']
        r_rate = -wr * float(da @ da)
        r_tau = -0.01 * float(qf @ qf)
        info = {}
        if task == 'walk':
            psi = math.atan2(R[1, 0], R[0, 0])
            cp, sp = math.cos(psi), math.sin(psi)
            vw = d.qvel[0:3]
            vf = cp * vw[0] + sp * vw[1]
            vl = -sp * vw[0] + cp * vw[1]
            wz = (R @ w)[2]
            cx, cw = self.cmd
            if abs(cx) > 1e-4:
                r_v = 3.0 * float(np.clip(vf / cx, -0.5, 1.2)) - 1.0 * abs(vf / cx - 1.0) * 0.5
            else:
                r_v = 2.0 * math.exp(-(vf / 0.01) ** 2)
            if abs(cw) > 1e-4:
                r_w = 1.5 * float(np.clip(wz / cw, -0.5, 1.2))
            else:
                r_w = 1.0 * math.exp(-(wz / 0.15) ** 2)
            r_pose = 0.0
            if abs(cx) <= 1e-4 and abs(cw) <= 1e-4:      # standing: return to the nominal pose
                qh = (d.qpos[8] ** 2 + d.qpos[9] ** 2) / self.hip_scale ** 2
                r_pose = -0.3 * qh - 0.3 * (d.qpos[7] / self.lift_scale) ** 2
            r = (0.3 + r_v + r_w + r_pose
                 - 8.0 * (1.0 - up_dot)
                 - 0.02 * float(w[0] ** 2 + w[1] ** 2)
                 - min(0.5 * (vl / 0.03) ** 2, 2.0)
                 + r_rate + r_tau)
            done = bool(up_dot < cfg['fall_up'] or d.qpos[2] < 0.035)
            if cfg.get('no_terminate'):
                done = False
            if done:
                r -= 10.0
            info.update(vf=vf, wz=wz)
            return r, done, info
        # ---- getup ----
        h = min(d.qpos[2] / HIP_H, 1.0)
        upright = up_dot
        nominal = (not cfg.get('getup_strict', True)) or ((abs(d.qpos[8]) < 0.35) and (abs(d.qpos[9]) < 0.35) and (abs(d.qpos[7]) < 0.012))
        stable = (up_dot > 0.97) and (d.qpos[2] > 0.066) and (np.linalg.norm(w) < 0.6) and nominal
        self.stand_time = self.stand_time + self.dt if stable else 0.0
        r_arms = 0.0
        if up_dot > 0.9:                      # once upright, bring the arms back to the nominal stance
            r_arms = -0.6 * (d.qpos[8] ** 2 + d.qpos[9] ** 2) / 0.36 - 0.3 * (d.qpos[7] / 0.014) ** 2
        r = (1.0 * upright + 1.0 * h + r_arms
             - 0.002 * float(w @ w)
             + r_rate + r_tau)
        done = False
        if self.stand_time > 1.0:
            r += 20.0
            done = True
            info['success'] = True
        return r, done, info

    def reset(self):
        self._init_state()
        obs = self._obs()
        return obs, self._priv(self.d.xmat[self.id_torso].reshape(3, 3))
