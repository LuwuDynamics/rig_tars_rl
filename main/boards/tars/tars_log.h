/*
 * tars_log.h - serial log / test tool for the Tars robot (Tars-only, lives in main/boards/tars/).
 *
 * What it is: a small text shell on the USB serial port (and UART0) that can
 *   - show a live status line (IMU, 3 servo positions / loads, servo bus voltage)
 *   - record IMU + servo read-back + servo commands at up to 100-200 Hz into RAM and send them to the PC as text
 *   - run servo step / sine tests and classic-gait walk / turn tests while recording
 *   - hold poses, switch the servos limp (move legs by hand), run the IMU axis calibration, switch the learned mode
 * Host side: tars_serial.py (see LOGGING.md).  Nothing happens until the host sends a command.
 *
 * Code layout: tars_log_core.* (events, ring, text format) and tars_log_shell.* (commands, sessions) are portable C99 and
 * unit-tested on the host; this header + tars_log.cc are the ESP32 glue (USB serial I/O, tasks, hooks into the robot).
 */
#ifndef TARS_LOG_H
#define TARS_LOG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* one 100 Hz IMU sample as produced by the tars_rl IMU task (only while the log tool / learned mode asked for the stream) */
typedef struct {
    float acc[3];          /* m/s^2, legacy imu.cc frame */
    float gyro[3];         /* deg/s, legacy imu.cc frame (x/y swapped with respect to the accelerometer) */
    float up[3];           /* attitude filter "up" vector in the robot frame; zeros while the filter is not running */
    uint8_t flags;         /* bit0 filter ready, bit1 IMU axis calibration stored, bit2 learned mode on, bit3 dry run, bit4 IMU missing, bit5 accelerometer axis stuck at full scale */
    int8_t state;          /* learned supervisor state, -1 when the learned mode is off */
} tars_log_imu_sample_t;

void tars_log_init(void);                          /* once, after imu_init() and tars_rl_init() */
void tars_error_log_init(void);                    /* once at boot, before starting board services */

/* robot_task side (robot.cc / robot_action.cc) */
bool tars_log_owns_bus(void);                      /* the tool needs servo read-back / holds the servos: call tars_log_bus_tick() each cycle instead of the slow polling */
void tars_log_bus_tick(void);
bool tars_log_owns_servos(void);                   /* a test / held pose / limp mode is active: the classic and learned gaits must not write the servos */

/* data hooks (any task) */
void tars_log_sample(const tars_log_imu_sample_t *s);
void tars_log_servo_fb(int id, int pos, int spd, int load);      /* a servo state frame was parsed (motor.cc) */
void tars_log_servo_volt(float volts);                           /* a servo bus voltage frame was parsed (motor.cc) */
void tars_log_servo_cmd(const short pos[3], short speed);        /* SetMotorPos() sent a new target (motor.cc) */

/* printf-style message for the host (printed as "# text", stored as a marker while recording). Never blocks. */
void tars_log_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* hold the servos at a pose (WritePos units, counts from the zero pose) for at most timeout_ms; the gaits stay quiet meanwhile */
void tars_log_hold_pose(const float pos_counts[3], unsigned timeout_ms);
void tars_log_release_servos(void);

#ifdef __cplusplus
}
#endif
#endif
