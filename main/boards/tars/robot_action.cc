#include "robot_action.h"
#include "tars_log.h"
#include "tars_rl.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <math.h>

static const char* TAG = "ROBOT_ACTION";

uint8_t ACTION_DONE = 0;

static int64_t gait_t0_us = 0;
static int64_t gait_until_us = 0;

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int clamp_to(int speed, int limit) {
    if (speed > limit) return limit;
    if (speed < -limit) return -limit;
    return speed;
}

// 曲柄滑块：滑块相对电机轴 x 固定，只沿 z 动。与 tars_ik.py 一致。
static constexpr float kCrankL1 = 26.0f;
static constexpr float kCrankL2 = 30.0f;
static constexpr float kCrankX = 13.7f;
static constexpr float kCrankZ0 = 33.1f;

static bool crank_theta(float z, float* out) {
    float r2 = kCrankX * kCrankX + z * z;
    float r = sqrtf(r2);
    if (r > kCrankL1 + kCrankL2 || r < fabsf(kCrankL1 - kCrankL2) || r < 1e-4f) {
        return false;
    }
    float cos_a = (kCrankL1 * kCrankL1 + r2 - kCrankL2 * kCrankL2) / (2.0f * kCrankL1 * r);
    if (cos_a > 1.0f) cos_a = 1.0f;
    if (cos_a < -1.0f) cos_a = -1.0f;
    *out = atan2f(kCrankX, z) + acosf(cos_a);
    return true;
}

// z 为相对零位 33.1 mm 的增量。返回 WritePos 用的第 3 轴偏移：
// WritePos 对该轴再乘 mask -1，最终等于 tars.py height() 的 -ik/300*1024。
static float lift_offset(float z_mm) {
    static float theta0 = 0.0f;
    static bool ready = false;
    if (!ready) {
        ready = crank_theta(kCrankZ0, &theta0);
    }
    float theta = 0.0f;
    if (!ready || !crank_theta(kCrankZ0 + z_mm, &theta)) {
        return 0.0f;
    }
    float deg = (theta - theta0) * (180.0f / (float)PI);
    return deg / 300.0f * (float)M_N;
}

float tars_lift_offset(float z_mm) {
    return lift_offset(z_mm);
}

void Reset() {
    vx = 0.0f;
    vyaw = 0.0f;
    Action_ID = 0;
    actionLoop_FLAG = 0;
    ACTION_DONE = 0;
    gait_t0_us = 0;
    gait_until_us = 0;
    float zero[MOTOR_NUM] = {0, 0, 0};
    WritePos(zero);
}

void tars_set_vx(int speed) {
    vx = (float)clamp_to(speed, (int)TARS_GAIT_VX_MAX);
    Action_ID = 0;
    control_mode = 0;
    motor_speed = 0;
    gait_until_us = 0;
}

void tars_set_vyaw(int speed) {
    vyaw = (float)clamp_to(speed, (int)TARS_GAIT_VYAW_MAX);
    Action_ID = 0;
    control_mode = 0;
    motor_speed = 0;
    gait_until_us = 0;
}

void tars_set_forward(int speed, int time_ms) {
    tars_set_vx(speed);
    vyaw = 0.0f;
    gait_t0_us = 0;
    gait_until_us = (time_ms > 0) ? (esp_timer_get_time() + (int64_t)time_ms * 1000) : 0;
    ESP_LOGI(TAG, "forward speed=%d time=%d", speed, time_ms);
}

void tars_set_turn(int speed, int time_ms) {
    tars_set_vyaw(speed);
    vx = 0.0f;
    gait_t0_us = 0;
    gait_until_us = (time_ms > 0) ? (esp_timer_get_time() + (int64_t)time_ms * 1000) : 0;
    ESP_LOGI(TAG, "turn speed=%d time=%d", speed, time_ms);
}

void tars_gait_update() {
    if (gait_until_us > 0 && esp_timer_get_time() >= gait_until_us) {
        vx = 0.0f;
        vyaw = 0.0f;
        gait_until_us = 0;
        gait_t0_us = 0;
    }

    // 串口日志/测试工具正在接管舵机（测试波形、保持姿态、卸力）：经典步态和学习型行为都不写舵机。
    if (tars_log_owns_servos()) {
        return;
    }

    // 学习型行为（行走反馈步态 + 摔倒检测 + 自动爬起 + 走直）。IMU 标定已存时开机默认自动开启（NVS auto=2）；可经 MCP self.tars.learned_mode 或串口 `learn off` 关闭。
    if (tars_rl_active()) {
        tars_rl_update(vx, vyaw);
        return;
    }

    float amp_x = clampf(vx, -TARS_GAIT_VX_MAX, TARS_GAIT_VX_MAX);
    float amp_yaw = clampf(vyaw, -TARS_GAIT_VYAW_MAX, TARS_GAIT_VYAW_MAX);

    const float kDead = 0.5f;
    if (fabsf(amp_x) < kDead && fabsf(amp_yaw) < kDead) {
        gait_t0_us = 0;
        float zero[MOTOR_NUM] = {0, 0, 0};
        WritePos(zero);
        return;
    }

    int64_t now = esp_timer_get_time();
    if (gait_t0_us == 0) {
        gait_t0_us = now;
    }
    float t = (now - gait_t0_us) / 1000000.0f;
    float phase = TARS_GAIT_FREQ_HZ * t * 2.0f * (float)PI;
    float c = cosf(phase);
    float s = sinf(phase);

    float pos[MOTOR_NUM] = {0, 0, 0};
    // 速度值就是两臂摆幅（舵机计数）。前进/后退最高 35，转向最高 25。
    // 身体起伏：一起步 ±9 mm，速度拉满 ±14 mm。
    float speed_ratio = 0.0f;
    if (fabsf(amp_yaw) >= fabsf(amp_x)) {
        speed_ratio = fabsf(amp_yaw) / TARS_GAIT_VYAW_MAX;
        pos[0] = s * amp_yaw;
        pos[1] = s * amp_yaw;
    } else {
        speed_ratio = fabsf(amp_x) / TARS_GAIT_VX_MAX;
        pos[0] = -s * amp_x;
        pos[1] = s * amp_x;
    }
    float lift_mm = TARS_GAIT_LIFT_MIN_MM +
                    (TARS_GAIT_LIFT_MAX_MM - TARS_GAIT_LIFT_MIN_MM) * speed_ratio;
    pos[2] = lift_offset(lift_mm * c);

    WritePos(pos);
}

void robot_action() {
    if (Action_ID == reset_ID) {
        Reset();
    } else {
        Action_ID = 0;
    }
}
