/*
 * tars_log_port.h - the few platform services the portable log shell (tars_log_shell.c) needs.
 *
 * Implemented by tars_log.cc on the robot and by the host-side simulator in tools/tars_log/test/ for the self test.
 */
#ifndef TARS_LOG_PORT_H
#define TARS_LOG_PORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tars_log_core.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t tlp_now_us(void);
void tlp_delay_ms(uint32_t ms);
void tlp_lock(void);                      /* very short critical section shared by the robot task and the log task */
void tlp_unlock(void);

/* Send text ('\n' terminated lines) to the host. May block briefly while the link is busy.
 * tlp_write returns the number of bytes accepted (less than n = the host is not reading right now: try again later);
 * tlp_out is the best-effort convenience for short messages (true = everything was accepted). */
size_t tlp_write(const char *s, size_t n);
bool tlp_out(const char *s, size_t n);
void tlp_set_link(int console);           /* 1 = answer through the console (stdout) path instead of writing the USB FIFO directly (fallback) */
bool tlp_set_baud(uint32_t baud);        /* change the serial rate of the console UART (the reply was queued before; it leaves at the old rate) */
void tlp_baud_confirm(void);              /* the PC followed the rate change: keep it (otherwise the device reverts to 115200 by itself) */
void tlp_flush_msgs(void);                /* print queued asynchronous messages now (keeps command replies in order) */

/* big event buffer (PSRAM first). Returns NULL if impossible; *cap is the number of events granted. */
tars_log_ev_t *tlp_alloc_events(uint32_t *cap);

/* robot access */
void tlp_imu_stream(bool on);             /* 100 Hz IMU task + attitude filter (also needed for the 'I' events) */
bool tlp_imu_ok(void);
int tlp_calibrating(void);                /* != 0 while the servo zero-calibration mode is active (servos are limp) */
bool tlp_imu_read(uint8_t reg, uint8_t *buf, size_t n);     /* QMI8658 register access for the 'imu' diagnostics (own I2C handle on the IMU bus) */
bool tlp_imu_write(uint8_t reg, uint8_t val);
int16_t tlp_zero_pos(int idx);            /* servo zero register, idx 0..2 */
int tlp_servo_mask(int idx);              /* +1 / -1 applied by WritePos */
void tlp_walk(int vx, int ms);            /* classic gait: forward/backward amplitude in counts, ms = 0 -> until stopped */
void tlp_turn(int vyaw, int ms);
void tlp_gait_stop(void);

/* learned-behaviour runtime (tars_rl) */
bool tlp_rl_calibrate(int step);
void tlp_rl_mode(bool enable, bool dry_run);
bool tlp_rl_is_calibrated(void);
void tlp_rl_map(bool flip, bool swap_lr);
void tlp_rl_map3(int sign_id1, int sign_id2, int sign_id3, bool swap_lr);   /* +1 = as hypothesis (a), -1 = that servo turns the other way */
bool tlp_rl_active(void);                 /* the learned mode is on (it drives the servos unless dry) */
void tlp_rl_selftest(int step);
void tlp_rl_status(char *buf, size_t n);
void tlp_rl_autostart(int mode);          /* 0 off, 1 stand + get up, 2 + walking commands; takes effect at the next power-up (persistent) */
int tlp_rl_autostart_get(void);
int tlp_rl_gait_set(int idx, bool save);  /* 0 v4, 1 v5, 2 v5 + faster forward, 3 v6 (foot plates), 4 v7 (smoother); returns 0, -1 bad index, -2 walking now */
int tlp_rl_gait_get(void);
int tlp_rl_getup_mode(int mode, bool save);   /* learned get-up: 0 keyframes, 1 network first, 2 network only; 0, -1 bad mode, -3 no network compiled in */
int tlp_rl_getup_mode_get(void);
int tlp_rl_getup_nn_available(void);
const char *tlp_rl_gait_name(int idx);    /* short text of a gait set ("v5", ...) */
int tlp_rl_gait_count(void);              /* number of gait sets: valid indices are 0 .. count-1 */
/* replace one gait ('f' 'b' 't' 'r') of the current set by 15 parameters in millionths (tars_gait_t order: q0 a1 b1 a2 b2 z0 c1 d1 c2 d2 f kp_h kd_h kp_z kd_z);
 * volatile; returns 0, -1 bad letter / out of range, -2 walking now */
int tlp_rl_gait_custom(char which, const int32_t micro[15]);
unsigned tlp_rl_gait_custom_mask(void);   /* bit 0 f, 1 b, 2 t, 3 r */
void tlp_rl_speed_map(int on);                   /* speed command on / off (the walking command's magnitude scales the gait amplitude) */
int tlp_rl_speed_map_get(void);
void tlp_rl_speed_max(int pct);                  /* top amplitude of the speed command in percent (70..100) */
int tlp_rl_speed_max_get(void);
void tlp_rl_straight(int on, int kp_pct, int kd_pct, int trim_pm);   /* walking straight: heading hold + trim of the forward / backward walks (volatile) */
void tlp_rl_straight_get(int out[4]);            /* on, kp_pct, kd_pct, trim_pm */
void tlp_rl_tune(int kp_pct, int kd_pct, int lpf_hz, int amp_pct, int rate_hz);   /* walking feedback tuning (volatile) */
void tlp_rl_tune_get(int out[5]);
bool tlp_rl_history(int idx, char *buf, size_t n);   /* idx 0 = oldest retained entry; false when there is no such entry */
/* show motions (nod, bow, headbang, the emotions): returns 0 accepted, -1 unknown name / speed outside 30..100, -2 learned mode off, -3 busy (not standing / another motion) */
int tlp_rl_motion_count(void);
const char *tlp_rl_motion_name(int idx);
int tlp_rl_motion(const char *name, int speed_pct);
void tlp_rl_motion_stop(void);

void tlp_info(char *buf, size_t n);       /* one line: build time, link, heap ... */
void tlp_error_log(void (*output)(const char *, ...)); /* retained errors, oldest first */
void tlp_reboot(void);

#ifdef __cplusplus
}
#endif
#endif
