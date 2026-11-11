#!/usr/bin/env python3
"""How much did the robot turn while it walked?  Integrates the recorded gyro about the vertical axis over the walking part of one or more recordings.

usage: python3 yaw_drift.py REST.tlog WALK1.tlog [WALK2.tlog ...]

REST.tlog is a recording (`rec 3`, or the standing part of any recording) of the robot standing still: its mean gyro is the zero-rate offset (the Tars IMUs have 6..10 deg/s of it).
The heading change is positive when the robot turned LEFT (counter-clockwise seen from above: the direction of a 'turn left' command; checked with recordings of
TURN_L / TURN_R: a left turn counts positive).  The same sign as the gyro z inside the firmware.  Each walk is reported as
  heading change [deg] over the walk (without the 1 s ramp-in), mean yaw rate [deg/s], duration, and the slowly varying yaw rate (>1.5 s average) range.
With `learn straight on` the heading change should stay within a few degrees; with it off a drift of 0.5..2 deg/s is typical."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tlog  # noqa: E402


def to_acc_frame(g):
    """the recorded gyro has the legacy x/y swap of imu.cc: swap back to get the accelerometer frame"""
    return np.stack([g[:, 1], g[:, 0], g[:, 2]], axis=1)


LEFT = -1.0     # the rate about the accelerometer's up axis is positive for a RIGHT turn on this robot (measured with TURN_L / TURN_R recordings): flip it


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rest = tlog.load(sys.argv[1])
    ghat = rest.acc.mean(axis=0)
    ghat = ghat / np.linalg.norm(ghat)                       # vertical axis in the accelerometer frame
    bias = to_acc_frame(rest.gyro).mean(axis=0)
    print("zero-rate offset (accelerometer frame, deg/s): %s | along the vertical axis: %+.2f   (heading change: + = left, - = right)" % (np.round(bias, 2), bias @ ghat))
    for path in sys.argv[2:]:
        rec = tlog.load(path)
        st, t = rec.imu_state, rec.imu_t
        walk = (st >= 1) & (st <= 2)                          # WALK_F, WALK_B
        if walk.sum() < 50:
            print("%-28s no forward / backward walk in this recording" % os.path.basename(path))
            continue
        t0, t1 = t[walk][0], t[walk][-1]
        m = walk & (t >= t0 + 1.0)
        yaw = LEFT * ((to_acc_frame(rec.gyro) - bias) @ ghat)
        ang = np.trapezoid(yaw[m], t[m])
        k = max(3, int(round(1.5 / np.median(np.diff(t)))))
        slow = np.convolve(yaw, np.ones(k) / k, mode="same")
        what = tlog.STATE_NAMES[int(st[walk][0])]
        print("%-28s %-7s %4.1f s: heading change %+6.1f deg | mean yaw rate %+5.2f deg/s | slow part %+5.1f .. %+5.1f deg/s" %
              (os.path.basename(path), what, t[m][-1] - t[m][0], ang, yaw[m].mean(), slow[m].min(), slow[m].max()))


if __name__ == "__main__":
    main()
