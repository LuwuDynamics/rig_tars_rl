"""The firmware's open-loop gait (robot_action.cc: tars_gait_update) re-implemented in Python."""
import math
import numpy as np

COUNT_DEG = 300.0 / 1024.0
GAIT_FREQ = 1.25
VX_MAX, VYAW_MAX = 35.0, 25.0
LIFT_MIN_MM, LIFT_MAX_MM = 9.0, 14.0


def gait_targets(t, vx=0.0, vyaw=0.0, hip_sign=(-1.0, +1.0)):
    """returns (q_hip_servo1_rad, q_hip_servo2_rad, lift_m) in the simulator's convention
    (hip q>0 = foot forward, lift>0 = slider/legs up).  vx, vyaw in servo counts like the firmware."""
    ax = float(np.clip(vx, -VX_MAX, VX_MAX))
    ay = float(np.clip(vyaw, -VYAW_MAX, VYAW_MAX))
    if abs(ax) < 0.5 and abs(ay) < 0.5:
        return 0.0, 0.0, 0.0
    ph = 2 * math.pi * GAIT_FREQ * t
    c, s = math.cos(ph), math.sin(ph)
    if abs(ay) >= abs(ax):
        ratio = abs(ay) / VYAW_MAX
        pos0, pos1 = s * ay, s * ay
    else:
        ratio = abs(ax) / VX_MAX
        pos0, pos1 = -s * ax, s * ax
    lift_mm = LIFT_MIN_MM + (LIFT_MAX_MM - LIFT_MIN_MM) * ratio
    # counts -> foot-forward angle. servo1 count<0 = forward, servo2 count>0 = forward  (see report)
    q1 = hip_sign[0] * pos0 * COUNT_DEG
    q2 = hip_sign[1] * pos1 * COUNT_DEG
    return math.radians(q1), math.radians(q2), lift_mm * c / 1000.0
