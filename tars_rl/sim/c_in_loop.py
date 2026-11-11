"""Drive the simulator with the *C firmware controller* (tars_ctrl.c + Mahony from tars_rl_core.c) through ctypes."""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW = os.path.join(os.path.dirname(ROOT), 'main', 'boards', 'tars')
WRAPPER_DIR = os.path.join(ROOT, 'firmware')
LIB_DIR = os.path.join(ROOT, '.build')
LIB = os.path.join(LIB_DIR, 'libtarsctrl.dylib' if sys.platform == 'darwin' else 'libtarsctrl.so')


def build():
    flags = ['-DWITH_GETUP_NN', '-DWITH_MOTIONS']
    if os.path.isfile(os.path.join(WRAPPER_DIR, 'generated', 'tars_rl_stand_weights.h')):
        flags.append('-DWITH_STAND')
    os.makedirs(LIB_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=LIB_DIR) as build_dir:
        output = os.path.join(build_dir, os.path.basename(LIB))
        subprocess.check_call([
            'cc', '-O2', '-std=c99', '-shared', '-fPIC', '-I', FW, *flags,
            os.path.join(WRAPPER_DIR, 'sim_wrapper.c'),
            os.path.join(FW, 'tars_ctrl.c'), os.path.join(FW, 'tars_rl_core.c'),
            '-lm', '-o', output,
        ])
        os.replace(output, LIB)


MOTION_NAMES = []                                            # the names in the C table tars_motions[] (filled in when the library is loaded)


class CController:
    NAMES = ['q0', 'a1', 'b1', 'a2', 'b2', 'z0', 'c1', 'd1', 'c2', 'd2', 'f', 'kp_h', 'kd_h', 'kp_z', 'kd_z']
    STATES = ['STAND', 'WALK_F', 'WALK_B', 'TURN_L', 'TURN_R', 'FALLEN', 'GETUP1', 'GETUP2A', 'GETUP2B', 'GETUP2C', 'FAILED', 'GETUP_NN', 'MOTION']

    def __init__(self, gaits, getup):
        """gaits: {'F': dict|None, 'B':..., 'T':...} (parameter dicts)   getup: (x[2K], K, tend, rate)"""
        build()
        self.lib = ctypes.CDLL(LIB)
        f = ctypes.POINTER(ctypes.c_float)
        self.lib.sim_step.restype = ctypes.c_int
        self.lib.sim_motion_start.restype = ctypes.c_int
        self.lib.sim_motion_end.restype = ctypes.c_int
        self.lib.sim_motion_length.restype = ctypes.c_float
        self.lib.sim_motion_name.restype = ctypes.c_char_p
        self.lib.sim_motion_face.restype = ctypes.c_char_p
        MOTION_NAMES[:] = [self.lib.sim_motion_name(i).decode() for i in range(self.lib.sim_motion_count())]
        self.lib.sim_motion_tilt_limit.restype = ctypes.c_float
        for i, k in enumerate('FBTR'):
            if gaits.get(k):
                arr = (ctypes.c_float * 15)(*[gaits[k][n] for n in self.NAMES])
                self.lib.sim_set_gait(i, arr)
        x, K, tend, rate = getup
        hip = (ctypes.c_float * K)(*[float(v) for v in np.clip(x[:K], -1, 1)])
        lift = (ctypes.c_float * K)(*[float(v) for v in np.clip(x[K:2 * K], -1, 1)])
        self.lib.sim_set_getup(K, ctypes.c_float(tend), ctypes.c_float(2.4), ctypes.c_float(0.014), ctypes.c_float(rate), hip, lift)
        self._acc = (ctypes.c_float * 3)()
        self._gyr = (ctypes.c_float * 3)()
        self._out = (ctypes.c_float * 3)()
        self._up = (ctypes.c_float * 3)()

    def reset(self, kp=2.0):
        self.lib.sim_reset(ctypes.c_float(kp))

    def set_heading(self, trim=0.0, kp=0.0, kd=0.0):
        """unequal hip strokes for walking straight: eps = trim + kp * heading + kd * yaw rate (takes effect at the next reset())"""
        self.lib.sim_set_heading(ctypes.c_float(trim), ctypes.c_float(kp), ctypes.c_float(kd))

    def set_getup_mode(self, mode):
        """learned get-up: 0 keyframes only, 1 network on the 1st / 3rd attempt, 2 network only (takes effect at the next reset(); needs the exported weights)"""
        self.lib.sim_set_getup_mode(int(mode))

    def motion_names(self):
        """the show motions compiled into the C supervisor (same order as tars_motions[])"""
        return MOTION_NAMES

    def motion_start(self, name, speed=1.0):
        """ask the C supervisor for a show motion; 0 = accepted, -1 unknown / not compiled in, -3 busy"""
        if name not in MOTION_NAMES:
            return -1
        return self.lib.sim_motion_start(MOTION_NAMES.index(name), ctypes.c_float(speed))

    def motion_stop(self):
        self.lib.sim_motion_stop()

    def motion_end(self):
        """how the last motion ended: 0 not yet, 1 played to the end, 2 stopped, 3 walking command, 4 fall, 5 request timed out"""
        return self.lib.sim_motion_end()

    def motion_face(self, name):
        return self.lib.sim_motion_face(MOTION_NAMES.index(name)).decode()

    def motion_length(self, name):
        return self.lib.sim_motion_length(MOTION_NAMES.index(name))

    def motion_tilt_limit(self, name):
        return self.lib.sim_motion_tilt_limit(MOTION_NAMES.index(name))

    def set_stand(self, on, scale=100):
        """switch the standing push-recovery network on / off (authority in percent); takes effect at the next reset()"""
        self.lib.sim_set_stand(int(bool(on)), int(scale))

    def set_kp(self, kp):
        self.lib.sim_set_kp(ctypes.c_float(kp))

    def imu(self, acc, gyr, dt=0.01):
        for i in range(3):
            self._acc[i] = acc[i]; self._gyr[i] = gyr[i]
        self.lib.sim_imu(self._acc, self._gyr, ctypes.c_float(dt))

    def step(self, gyr, dt, vx, vyaw):
        for i in range(3):
            self._gyr[i] = gyr[i]
        st = self.lib.sim_step(self._gyr, ctypes.c_float(dt), ctypes.c_float(vx), ctypes.c_float(vyaw), self._out, self._up)
        return self._out[0], self._out[1], self._out[2], st
