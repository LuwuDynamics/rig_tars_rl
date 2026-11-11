/*
 * tars_log_shell.h - portable command shell, session state machines and servo test driver of the Tars log tool.
 *
 * Threads that call into it:
 *   log task    : tars_log_shell_feed() / tars_log_shell_poll()           (single thread, the only one that touches sessions)
 *   robot task  : tars_log_shell_tick()                                    (every robot_control() cycle)
 *   any task    : the tars_log_shell_on_*() data hooks, hold_pose / release
 */
#ifndef TARS_LOG_SHELL_H
#define TARS_LOG_SHELL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* what the robot task must do on this cycle (the shell never touches the servo bus itself) */
typedef struct {
    int write;             /* 1: WritePos(pos) - counts relative to the calibrated zero */
    float pos[3];
    int enable_id;         /* 1..3: EnableMotor(enable_id, enable_val) (torque on/off) */
    int enable_val;
    int read_id;           /* 1..3: ReadMotorState(read_id) */
    int read_volt;         /* 1: ReadServoVoltage(1) */
    int cmd_src;           /* tag for the 'C' event of this write (1 = log tool) */
} tars_log_tick_out_t;

typedef struct {
    int16_t acc[3];        /* raw LSB, 9.8/16384 m/s^2 (legacy imu.cc frame) */
    int16_t gyro[3];       /* raw LSB, 1/32 deg/s (legacy imu.cc frame: x/y swapped w.r.t. the accelerometer) */
    int16_t up[3];         /* attitude filter "up" vector in the robot frame, x10000 (0 until the filter runs) */
    uint8_t flags;         /* bit0 filter ready, bit1 IMU axis calibration present, bit2 learned mode on, bit3 dry run, bit4 IMU missing, bit5 an accelerometer axis is stuck at full scale */
    int8_t state;          /* learned supervisor state, -1 when the learned mode is off */
    int8_t vx, vyaw;       /* high-level walk commands currently set */
} tars_log_imu_t;

void tars_log_shell_init(void);
void tars_log_shell_feed(const uint8_t *data, size_t n);   /* bytes typed by the host */
bool tars_log_shell_poll(void);                            /* periodic work; returns true while there is more to do soon */
bool tars_log_shell_idle(void);                            /* no session (watch / rec / live / dump / scan) is running */
void tars_log_shell_say(const char *text);                 /* print "# text" */

bool tars_log_shell_bus_active(void);                      /* the shell wants the servo read-back / owns the servos -> call _tick every cycle */
bool tars_log_shell_owns_servos(void);                     /* classic + learned gait must stay away from the servos */
void tars_log_shell_tick(int64_t now_us, tars_log_tick_out_t *out);
void tars_log_shell_abort_servo_use(void);                 /* e.g. the servo calibration mode started */

void tars_log_shell_hold_pose(const float pos[3], uint32_t timeout_ms);
void tars_log_shell_release(void);

void tars_log_shell_on_imu(const tars_log_imu_t *s, int64_t now_us);
void tars_log_shell_on_servo(int id, int16_t pos, int16_t spd, int16_t load, int64_t now_us);
void tars_log_shell_on_cmd(const int16_t pos[3], int16_t speed, int src, int64_t now_us);
void tars_log_shell_on_volt(int16_t centivolt, int64_t now_us);
void tars_log_shell_on_msg(const char *text, int64_t now_us);

#ifdef __cplusplus
}
#endif
#endif
