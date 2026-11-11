/*
 * tars_rl.h - ESP32-S3 glue for the learned Tars behaviours: walking gaits with the gyro heading hold, fall detection, get-up (learned network + keyframes).
 *
 * Lives in main/boards/tars/ (Tars-only; compiled only for CONFIG_BOARD_TYPE_TARS).
 * Power-up state: the learned mode starts by itself ~4 s after boot when the IMU calibration is stored (autostart, see below); otherwise the classic gait is
 * untouched.  Switch it with the MCP tool "self.tars.learned_mode" or the log tool (`learn on go` / `learn off`; dry_run first on a new robot!).
 */
#ifndef TARS_RL_H
#define TARS_RL_H

#include <stdbool.h>
#include <stddef.h>
#include <esp_err.h>

#ifdef __cplusplus
extern "C" {
#endif

void tars_rl_init(void);                          /* once, after imu_init() / InitZeroPos() */
bool tars_rl_active(void);                        /* learned mode is driving the servos  */
bool tars_rl_owns_imu(void);                      /* learned mode OR the logger reads the IMU at 100 Hz itself (motor_rx_task must not) */
void tars_rl_imu_stream(bool on);                 /* logger: start/stop the 100 Hz IMU task + attitude filter without learned mode */
void tars_rl_set_mode(bool enable, bool dry_run); /* dry_run: compute + log, do NOT move the servos */
void tars_rl_start_after_calibration(void);       /* after both calibrations are saved, apply the configured autostart mode */
void tars_rl_update(float vx_counts, float vyaw_counts);   /* called from tars_gait_update() on every robot_task tick (2 ms); runs the controller at its own rate (500 Hz default) */

/* IMU axis calibration (accel frame -> robot frame).  step 0: hold upright & still;  step 1: tilt NOSE DOWN ~30 deg & still. */
bool tars_rl_calibrate_imu(int step);
bool tars_rl_calibrate_imu_upright(void);         /* known TARS mounting, robot upright and still */
bool tars_rl_is_calibrated(void);
esp_err_t tars_rl_erase_calibration(void);        /* remove saved IMU data before restarting */

/* servo mapping (persisted in NVS): flip=false -> hypothesis (a) default, true -> (b) all three servo signs flipped;
 * swap_lr=true -> ID2 is the LEFT arm (default: ID1 is left) */
void tars_rl_set_mapping(bool flip, bool swap_lr);
/* the general form: one sign per servo (+1 = as hypothesis (a), -1 = that servo turns the other way) */
void tars_rl_set_mapping3(int sign_id1, int sign_id2, int sign_id3, bool swap_lr);

/* servo direction self-test (hold the robot in the air!): 1 = hips foot-forward, 2 = lift legs up, 3 = zero pose, 0 = release.
 * The pose is held for 30 s by the log tool (tars_log_hold_pose), which keeps the gaits away from the servos. */
void tars_rl_selftest(int step);

/* Autostart (persisted in NVS, takes effect at the next power-up): 0 = off, 1 = the learned mode starts by itself ~4 s after power-up:
 * the robot stands, notices a fall and gets up; walking commands (BLE / voice) are ignored,  2 = the same, walking commands are accepted (the default
 * when nothing is stored).  It only fires when the IMU calibration was already stored at power-up: a robot that has not been calibrated stays passive. */
void tars_rl_set_autostart(int mode);
int tars_rl_autostart(void);

/* walking gait set (default 6 = "v9" when nothing is stored): 0 = "v4" (searched in the original simulator), 1 = "v5" (searched in the simulator calibrated with the real robot),
 * 2 = "v5" with a faster (less robust) forward gait, 3 = "v6" for the robot WITH 10 mm foot plates, 4 = "v7" (calibrated simulator, body-shaking
 * penalty, slower steps), 5 = "v8" (v7 forward + the smooth backward gait tested on the real robot), 6 = "v9" (sinusoidal hip patterns, built for the 500 Hz controller rate).  Returns 0, -1 (bad index) or -2 (a walking state is running: stop first).  save = keep the choice over power cycles. */
int tars_rl_set_gait_set(int idx, bool save);
int tars_rl_gait_set(void);
/* learned get-up (a neural network trained with PPO in simulation, tars_rl/REPORT.md section 21): mode 0 = keyframes only, 1 = network on the 1st and 3rd attempt and
 * keyframes on the 2nd (the default when nothing is stored), 2 = network only.  Effective at the next get-up attempt.  save = keep over power cycles.
 * Returns 0, -1 (bad mode), -3 (no network compiled in). */
int tars_rl_set_getup_mode(int mode, bool save);
int tars_rl_getup_mode(void);
int tars_rl_getup_nn_available(void);
const char *tars_rl_gait_name(int idx);   /* "v4", "v5", ... ("?" for a bad index) */
int tars_rl_gait_count(void);             /* number of gait sets: valid indices are 0 .. count-1 */
/* replace one gait of the current set by 15 parameters (order of tars_gait_t) from the serial line; volatile.  which = 'f' 'b' 't' 'r'.
 * Returns 0, -1 (bad letter / parameter outside the search ranges), -2 (a walking state is running). */
int tars_rl_set_custom_gait(char which, const float p[15]);
unsigned tars_rl_gait_custom_mask(void);  /* bit 0 f, 1 b, 2 t, 3 r: slots currently replaced */
/* speed command (volatile, ON at every boot): the magnitude of the walking command (servo counts, the MCP / BLE speed parameter) scales the gait amplitude:
 * 70 % at <= 14 counts ... 90 % at >= 20 counts (set with tars_rl_set_speed_max).  Off: always the full gait. */
void tars_rl_set_speed_map(int on);
int tars_rl_speed_map(void);
void tars_rl_set_speed_max(int pct);      /* top amplitude (command >= 20 counts) in percent, 70..100, volatile; default 90 */
int tars_rl_speed_max(void);
/* walking straight (volatile; ON at every boot): forward / backward walks keep their heading with a small left / right stroke difference, driven by the gyro (heading hold),
 * plus an optional constant correction.  kp_pct: stroke fraction x100 per rad of heading error (default 150), kd_pct: per rad/s of yaw rate (default 0),
 * trim_pm: constant correction in per mille of the stroke, + = turn right in both directions (default 0).  Limited to +-30 % of the stroke. */
void tars_rl_set_straight(int on, int kp_pct, int kd_pct, int trim_pm);
void tars_rl_get_straight(int out[4]);   /* on, kp_pct, kd_pct, trim_pm */

/* runtime tuning of the walking feedback (not stored: the searched gait is back at every power-up).  Percent values; defaults 100 100 0 100 500.
 * kp_pct: pitch feedback, kd_pct: pitch-rate feedback, lpf_hz: low-pass on the pitch rate (0 = off), amp_pct: periodic gait pattern,
 * rate_hz: controller rate, 50 (as simulated), 100 (10 ms less loop delay), 250 or 500 (the classic gait's rate: the servo commands change in fine steps;
 * the IMU / attitude filter stays at 100 Hz, so only the gait pattern and the command stream get finer). */
void tars_rl_set_tune(int kp_pct, int kd_pct, int lpf_hz, int amp_pct, int rate_hz);
void tars_rl_get_tune(int out[5]);

/* show motions (tars_rl/FUN_MOTIONS.md): nod, bow, headbang and the emotions - the robot performs a pre-computed motion where it stands and ends in the stand pose (about 6..8 s at full speed).
 * Needs the learned mode ON (a dry run only plays the state machine, the servos do not move).  tars_rl_motion() returns 0 = accepted (it starts as soon as the robot stands still
 * and upright; it waits up to 3 s for a walk to end - stop the walking command first), -1 = unknown motion or speed outside 30..100 %, -2 = the learned mode is off (or walking
 * commands are disabled, autostart mode 1), -3 = busy (falling, getting up, or another motion).  A walking command, a fall, tars_rl_motion_stop() or switching the learned mode
 * off ends a motion at once (the hips go home first, then the lift). */
int tars_rl_motion_count(void);
const char *tars_rl_motion_name(int idx);   /* "nod", "bow", ... ("?" for a bad index) */
int tars_rl_motion_find(const char *name);  /* index, or -1 */
int tars_rl_motion(int idx, int speed_pct);
void tars_rl_motion_stop(void);
const char *tars_rl_motion_face(int idx);   /* the screen emoji that goes with an emotion ("happy", "sad", ...), NULL for the plain show motions */
int tars_rl_motion_ms(int idx, int speed_pct);   /* how long the motion takes at that speed [ms], 0 for a bad index */
/* the glue tells the board when a motion with a face starts (face = the emoji name) and when it is over (face = "neutral"): called from the control task,
 * so the callback must not block (schedule the display call on another task). */
void tars_rl_set_face_cb(void (*cb)(void *ctx, const char *face), void *ctx);

/* why the robot restarted last time (power-on / BROWNOUT / PANIC / watchdog ...) */
const char *tars_reset_reason_name(void);

/* what the learned mode did (RAM ring, 160 entries, oldest first): state changes, mode on/off, autostart.  idx 0 = oldest retained. */
int tars_rl_history_count(void);
bool tars_rl_history(int idx, char *buf, size_t n);

/* one-line status for logs / MCP */
void tars_rl_status(char *buf, size_t n);

#ifdef __cplusplus
}
#endif
#endif
