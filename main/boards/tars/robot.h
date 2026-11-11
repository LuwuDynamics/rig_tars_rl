#ifndef __ROBOT_H
#define __ROBOT_H
#include <atomic>
#include "motor.h"

#define FLASH_ZERO_POS_ADDR 0xFFF000
#define PI 3.14159
#define TARS_GAIT_FREQ_HZ     1.25f   // 周期 0.8s，与 tars.py 一致
#define TARS_GAIT_VX_MAX      35.0f   // 前进/后退摆臂幅度（舵机计数）
#define TARS_GAIT_VYAW_MAX    25.0f   // 左转/右转摆臂幅度（舵机计数）
#define TARS_GAIT_LIFT_MIN_MM 9.0f    // 一起步就有的身体起伏 (mm)
#define TARS_GAIT_LIFT_MAX_MM 14.0f   // 速度拉满时的身体起伏 (mm)

void InitZeroPos();
bool WriteZeroPos();
bool ReadZeroPos();
bool FinishCalibration();
void RequestCalibrationCompletion();
void robot_control();
void set_action_loop_flag(uint8_t flag);

extern float vx;
extern float vyaw;
extern uint16_t motor_speed;
extern std::atomic<int> calibrate_mode;
extern uint8_t Action_ID;
extern uint8_t ACTION_DONE;
extern uint8_t actionLoop_FLAG;
extern int control_mode;
extern uint8_t isIMUInit;
extern float q_head;
extern char g_calib_error_msg[48];

void lulu_ble_on_rx_bytes(const uint8_t* data, size_t len);

#endif
