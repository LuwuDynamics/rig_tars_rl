/*
 * tars_rl_core.h - portable C99 runtime for the Tars RL policies (no ESP-IDF dependency).
 *
 * Mirrors sim/tars_env.py exactly:
 *   - Mahony attitude filter (accel + gyro)            -> tars_mahony_*
 *   - observation layout (28 floats)                   -> tars_policy_step
 *   - firmware open-loop gait used as the residual base -> tars_baseline_action
 *   - 2 x tanh MLP actor                               -> tars_mlp_forward
 *
 * Robot frame (same as the simulator):  X forward, Y left, Z up.
 *   hip angle   q > 0  : foot moves FORWARD   [rad]
 *   lift        z > 0  : hip frame / legs move UP relative to the torso [m]
 */
#ifndef TARS_RL_CORE_H
#define TARS_RL_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TARS_OBS_FRAME 12
#define TARS_OBS_DIM   28
#define TARS_ACT_DIM   3
#define TARS_COUNT_RAD 0.0051132692f   /* one SCS0009 count = 300/1024 deg */

/* ------------------------------------------------------------------ MLP */
typedef struct {
    int n_in, n_h1, n_h2, n_out;
    const float *w1, *b1;   /* [n_h1][n_in]   (obs normalisation already folded in) */
    const float *w2, *b2;   /* [n_h2][n_h1] */
    const float *w3, *b3;   /* [n_out][n_h2] */
    const float *lo, *hi;   /* per-input clamp = training mean -/+ 10 sigma (same as the +-10 clip on normalised obs) */
} tars_mlp_t;

/* out is clipped to [-1, 1].  Scratch: n_h1 + n_h2 floats on the stack (<= 2 KB). */
void tars_mlp_forward(const tars_mlp_t *m, const float *in, float *out);
/* The same with the scratch in static memory instead of the stack (2.3 KB): for the firmware tasks with small stacks (robot_task has 4 KB).  Not re-entrant:
 * call it from one task only. */
void tars_mlp_forward_static(const tars_mlp_t *m, const float *in, float *out);

/* ------------------------------------------------------------------ attitude filter */
typedef struct {
    float q[4];     /* w x y z, body->world */
    float kp;       /* accel correction gain [1/s] */
    float sig_a;    /* accel gating sigma [m/s^2] */
} tars_mahony_t;

void tars_mahony_init(tars_mahony_t *f, float kp);
void tars_mahony_init_from_acc(tars_mahony_t *f, const float acc[3]);
void tars_mahony_update(tars_mahony_t *f, const float gyro[3], const float acc[3], float dt);
/* unit vector "world up" expressed in the body frame (= measured accel direction at rest) */
void tars_mahony_up_body(const tars_mahony_t *f, float up[3]);

/* Build the IMU-to-body rotation from up and forward references in m/s². */
int tars_imu_axes_from_refs(const float up[3], const float forward[3], float rotation[9]);

/* ------------------------------------------------------------------ gyro zero-rate offset
 * The Tars IMUs have a steady gyro offset of 0.1..0.17 rad/s (the simulator randomises 0.012): it is estimated while the robot is
 * still, subtracted from every sample, and follows its slow drift (temperature) only while the robot stays static.
 * Not part of the training model (firmware only), hence not mirrored in sim/tars_env.py. */
typedef struct {
    float bias[3];                 /* rad/s, same frame as the samples that are fed in */
    float sum[3], sum2[3], acc[3]; /* start window: sum / sum of squares of the gyro, sum of the accelerometer */
    int n;                         /* samples in the start window */
    int static_n;                  /* consecutive static samples (drift following starts after 100 = 1 s at 100 Hz) */
} tars_gyro_bias_t;

void tars_gyro_bias_reset(tars_gyro_bias_t *b);
/* Feed raw 100 Hz samples until it returns 1: the robot was really still for 30 samples (|a| within 1.5 m/s^2 of g, every axis of the
 * gyro quiet: std < 0.05 rad/s).  Then bias[] is set and mean_acc_out[] holds the averaged accelerometer (initial attitude). */
int tars_gyro_bias_start(tars_gyro_bias_t *b, const float acc[3], const float gyro[3], float mean_acc_out[3]);
/* Running phase: subtracts the offset from gyro[] in place and lets it follow the drift while static. */
void tars_gyro_bias_apply(tars_gyro_bias_t *b, const float acc[3], float gyro[3]);

/* ------------------------------------------------------------------ policy wrapper */
typedef struct {
    /* configuration (must match training: see model/policy_*.json) */
    float hip_scale;       /* rad   (0.6)   */
    float lift_scale;      /* m     (0.014) */
    float res_hip;         /* rad   residual authority (0.25)  */
    float res_lift;        /* m     (0.008)                    */
    float gait_freq;       /* Hz    (1.25)                     */
    float amp_per_vx;      /* counts per m/s (1000)            */
    float amp_per_wz;      /* counts per rad/s (30)            */
    float yaw_sign;        /* -1                                */
    int   residual;        /* 1: residual on firmware gait, 0: direct */
    float act_slew;        /* max change of the normalised action per step, 0 = unlimited (training option act_slew of sim/tars_env.py) */
    /* state */
    float phase;           /* 0..1 */
    float a_prev[3];       /* previous total (normalised) action */
    float prev_frame[TARS_OBS_FRAME];
    int   have_prev;
    float cmd[2];          /* vx [m/s], wz [rad/s] */
} tars_policy_state_t;

void tars_policy_defaults(tars_policy_state_t *s, int residual);
void tars_policy_reset(tars_policy_state_t *s);

/* firmware gait (robot_action.cc: tars_gait_update) as normalised action */
void tars_baseline_action(const tars_policy_state_t *s, float a[3]);

/*
 * One 50 Hz policy step.
 *   up_body : filtered "up" vector in the body frame (tars_mahony_up_body)
 *   gyro    : body rates [rad/s] (latest sample, already in robot frame)
 *   q_meas  : measured {hip_l, hip_r [rad], lift [m]}
 *   dt      : 0.02
 * Outputs:
 *   action_out : total normalised action in [-1,1] (becomes a_prev next step)
 *   target_out : {hip_l, hip_r [rad], lift [m]} to command
 */
void tars_policy_step(tars_policy_state_t *s, const tars_mlp_t *mlp,
                      const float up_body[3], const float gyro[3], const float q_meas[3], float dt,
                      float action_out[3], float target_out[3]);

/* ------------------------------------------------------------------ standing assist (push recovery network)
 * One implementation for the firmware glue and the C-in-the-loop simulator: the network sees the attitude, the gyro and the previous COMMANDED
 * targets (training option obs_q_mode 'cmd': no servo read-back needed) and decides the hip / lift targets while the robot stands. */
typedef struct {
    tars_policy_state_t pol;
    float prev[3];             /* last commanded {hip_l, hip_r [rad], lift [m]}, rounded to whole servo counts */
    int   scale_pct;           /* authority 10..100 %: the network's output is scaled before it is commanded */
} tars_stand_t;

void tars_stand_init(tars_stand_t *s, int residual, float act_slew);
void tars_stand_reset(tars_stand_t *s);                 /* forget the memory (called when the assist (re)starts) */
void tars_stand_step(tars_stand_t *s, const tars_mlp_t *mlp, const float up[3], const float gyro[3], float dt, float q[2], float *z);

/* build the raw (un-normalised) observation vector only - used by tests */
void tars_build_obs(const tars_policy_state_t *s, const float up_body[3], const float gyro[3],
                    const float q_meas[3], float obs[TARS_OBS_DIM]);

#ifdef __cplusplus
}
#endif
#endif
