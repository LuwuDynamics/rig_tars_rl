/* Thin C wrapper so the *firmware* controller (tars_ctrl.c + tars_rl_core.c Mahony) can be driven from the
 * Python simulator through ctypes ("C-in-the-loop").  Not part of the ESP32 build. */
#include <stddef.h>
#include <string.h>
#include "tars_ctrl.h"
#include "tars_rl_core.h"

static tars_ctrl_t C;
static tars_mahony_t MF;
static int mf_ready;
static tars_gait_t G[4];
static int g_set[4];
static float GH[64], GL[64];
static tars_getup_t GU;

/* standing push-recovery network (only when the exported weights are present: -DWITH_STAND, generated/tars_rl_stand_weights.h) */
static int g_stand_on, g_stand_scale = 100;
#ifdef WITH_STAND
#include "generated/tars_rl_stand_weights.h"
static tars_stand_t SA;
static void sa_reset(void *ctx) { (void)ctx; tars_stand_reset(&SA); }
static void sa_step(void *ctx, const float up[3], const float gyro[3], float dt, float q[2], float *z)
{
    (void)ctx;
    tars_stand_step(&SA, &tars_stand_mlp, up, gyro, dt, q, z);
}
#endif

/* Learned get-up weights from the TARS board sources. */
static int g_getup_mode;
#ifdef WITH_GETUP_NN
#include "tars_rl_getup_nn_weights.h"
static void gn_cb(void *ctx, const float obs[28], float act[3]) { (void)ctx; tars_mlp_forward(&tars_getup_nn_mlp, obs, act); }
#endif
void sim_set_getup_mode(int mode) { g_getup_mode = mode; }      /* effective from the next sim_reset */

/* Motion tables from the TARS board sources. */
#ifdef WITH_MOTIONS
#include "tars_motions.h"
int sim_motion_count(void) { return TARS_MOTION_COUNT; }
const char *sim_motion_name(int idx) { return idx >= 0 && idx < TARS_MOTION_COUNT ? tars_motions[idx].name : ""; }
const char *sim_motion_face(int idx) { return idx >= 0 && idx < TARS_MOTION_COUNT && tars_motions[idx].face ? tars_motions[idx].face : ""; }
int sim_motion_start(int idx, float speed) { return idx >= 0 && idx < TARS_MOTION_COUNT ? tars_ctrl_motion_start(&C, &tars_motions[idx], speed) : -1; }
float sim_motion_tilt_limit(int idx) { return idx >= 0 && idx < TARS_MOTION_COUNT ? tars_motions[idx].tilt_limit_deg : -1.0f; }
float sim_motion_length(int idx) { return idx >= 0 && idx < TARS_MOTION_COUNT ? (float)(tars_motions[idx].n - 1) * tars_motions[idx].dt : -1.0f; }
#else
int sim_motion_count(void) { return 0; }
const char *sim_motion_name(int idx) { (void)idx; return ""; }
const char *sim_motion_face(int idx) { (void)idx; return ""; }
int sim_motion_start(int idx, float speed) { (void)idx; (void)speed; return -1; }
float sim_motion_tilt_limit(int idx) { (void)idx; return -1.0f; }
float sim_motion_length(int idx) { (void)idx; return -1.0f; }
#endif
void sim_motion_stop(void) { tars_ctrl_motion_stop(&C); }
int sim_motion_end(void) { return C.motion_end; }

static float g_trim, g_hold_kp, g_hold_kd;
void sim_set_heading(float trim, float kp, float kd)      /* effective from the next sim_reset */
{
    g_trim = trim; g_hold_kp = kp; g_hold_kd = kd;
}

void sim_set_stand(int on, int scale)         /* effective from the next sim_reset */
{
    g_stand_on = on;
    g_stand_scale = scale;
}

void sim_set_gait(int which, const float *p)   /* 15 floats in the struct order */
{
    memcpy(&G[which], p, sizeof(tars_gait_t));
    g_set[which] = 1;
}

void sim_set_getup(int k, float tend, float hip_range, float lift_range, float rate, const float *hip, const float *lift)
{
    memcpy(GH, hip, sizeof(float) * k);
    memcpy(GL, lift, sizeof(float) * k);
    GU.k = k; GU.tend = tend; GU.hip_range = hip_range; GU.lift_range = lift_range; GU.rate = rate;
    GU.hip = GH; GU.lift = GL;
}

void sim_reset(float kp)
{
    tars_ctrl_init(&C);
    tars_ctrl_reset(&C);
    C.gait_f = g_set[0] ? &G[0] : NULL;
    C.gait_b = g_set[1] ? &G[1] : NULL;
    C.gait_turn = g_set[2] ? &G[2] : NULL;
    C.gait_turn_r = g_set[3] ? &G[3] : NULL;
    C.getup = &GU;
    C.straight = (g_trim != 0.0f || g_hold_kp != 0.0f || g_hold_kd != 0.0f);
    C.trim = g_trim; C.hold_kp = g_hold_kp; C.hold_kd = g_hold_kd;
#ifdef WITH_STAND
    tars_stand_init(&SA, TARS_STAND_RESIDUAL, TARS_STAND_ACT_SLEW);
    SA.scale_pct = g_stand_scale;
    C.stand_reset = sa_reset;
    C.stand_step = sa_step;
    C.stand_on = g_stand_on;
#endif
#ifdef WITH_GETUP_NN
    C.getup_nn_fn = gn_cb;
    C.getup_nn_mode = g_getup_mode;
#endif
    tars_mahony_init(&MF, kp);
    mf_ready = 0;
}

void sim_imu(const float *acc, const float *gyro, float dt)
{
    if (!mf_ready) { tars_mahony_init_from_acc(&MF, acc); mf_ready = 1; }
    tars_mahony_update(&MF, gyro, acc, dt);
}

/* out: q_left, q_right, lift ; returns state id */
int sim_step(const float *gyro, float dt, float vx, float vyaw, float *out, float *up_out)
{
    float up[3], q[2], z;
    tars_mahony_up_body(&MF, up);
    tars_ctrl_step(&C, up, gyro, dt, vx, vyaw, q, &z);
    out[0] = q[0]; out[1] = q[1]; out[2] = z;
    if (up_out) { up_out[0] = up[0]; up_out[1] = up[1]; up_out[2] = up[2]; }
    return (int)C.state;
}

void sim_set_kp(float kp) { MF.kp = kp; }
