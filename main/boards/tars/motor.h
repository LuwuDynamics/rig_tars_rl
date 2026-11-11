#ifndef __MOTOR_H
#define __MOTOR_H

/* Tars 组态：3 × SCS009。协议见 boards/common/motors/scs009.h */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <driver/uart.h>

#define MOTOR_NUM 3
#define M_N 1024
#define M_A 360.0
#define STALL_POS_THRESHOLD   100
#define STALL_TOR_THRESHOLD   100
#define STALL_DEBOUNCE_COUNT  3
#define STALL_COOLDOWN_MS     1000

typedef struct
{
    uint8_t ID;
    short DesPos;
    float DesSpd;
    short DesTor;
    short FbPos;
    float FbSpd;
    short FbTor;
    short ZeroPos;
    uint8_t Load;
    int64_t FbTimeUs;   // esp_timer time (us) when the last state frame of this servo was parsed (logger)
} Motor;

typedef void (*motor_stall_callback_t)(uint8_t motor_id);

void SendMotorCommand(uint8_t *pData, uint16_t size);
void SetMotorPos(short pos[], short vel);
void SetMotorAngle(float angle[], short vel);
void WritePos(const float pos[MOTOR_NUM]);
int MotorMask(int idx);   // +1 / -1 applied by WritePos() to servo idx (0..2)
void EnableMotor(uint8_t ID, uint8_t mode);
void EnableAllMotor(int mode);
void ReadMotorState(uint8_t ID);
void ReadServoVoltage(uint8_t ID);
void EnableStallDetection(bool enable);
void motor_rx();

extern Motor motor[MOTOR_NUM];
extern float servo_voltage;
extern int64_t servo_voltage_time_us;   // when servo_voltage was last updated (logger)

#endif
