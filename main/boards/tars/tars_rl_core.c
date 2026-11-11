/* tars_rl_core.c - see tars_rl_core.h.  Keep in lock-step with sim/tars_env.py. */
#include "tars_rl_core.h"

#include <math.h>
#include <string.h>

#define TARS_G 9.81f
#define TARS_2PI 6.28318530718f

static float clampf_(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ------------------------------------------------------------------ MLP */
void tars_mlp_forward(const tars_mlp_t *m, const float *in, float *out)
{
    float x[64], h1[256], h2[256];
    const int n1 = m->n_h1, n2 = m->n_h2;

    for (int j = 0; j < m->n_in; j++) x[j] = m->lo ? clampf_(in[j], m->lo[j], m->hi[j]) : in[j];
    for (int i = 0; i < n1; i++) {
        const float *w = m->w1 + (size_t)i * m->n_in;
        float acc = m->b1[i];
        for (int j = 0; j < m->n_in; j++) acc += w[j] * x[j];
        h1[i] = tanhf(acc);
    }
    for (int i = 0; i < n2; i++) {
        const float *w = m->w2 + (size_t)i * n1;
        float acc = m->b2[i];
        for (int j = 0; j < n1; j++) acc += w[j] * h1[j];
        h2[i] = tanhf(acc);
    }
    for (int i = 0; i < m->n_out; i++) {
        const float *w = m->w3 + (size_t)i * n2;
        float acc = m->b3[i];
        for (int j = 0; j < n2; j++) acc += w[j] * h2[j];
        out[i] = clampf_(acc, -1.0f, 1.0f);
    }
}

static float s_x[64], s_h1[256], s_h2[256];

void tars_mlp_forward_static(const tars_mlp_t *m, const float *in, float *out)
{
    float *x = s_x, *h1 = s_h1, *h2 = s_h2;
    const int n1 = m->n_h1, n2 = m->n_h2;

    for (int j = 0; j < m->n_in; j++) x[j] = m->lo ? clampf_(in[j], m->lo[j], m->hi[j]) : in[j];
    for (int i = 0; i < n1; i++) {
        const float *w = m->w1 + (size_t)i * m->n_in;
        float acc = m->b1[i];
        for (int j = 0; j < m->n_in; j++) acc += w[j] * x[j];
        h1[i] = tanhf(acc);
    }
    for (int i = 0; i < n2; i++) {
        const float *w = m->w2 + (size_t)i * n1;
        float acc = m->b2[i];
        for (int j = 0; j < n1; j++) acc += w[j] * h1[j];
        h2[i] = tanhf(acc);
    }
    for (int i = 0; i < m->n_out; i++) {
        const float *w = m->w3 + (size_t)i * n2;
        float acc = m->b3[i];
        for (int j = 0; j < n2; j++) acc += w[j] * h2[j];
        out[i] = clampf_(acc, -1.0f, 1.0f);
    }
}

/* ------------------------------------------------------------------ Mahony */
void tars_mahony_init(tars_mahony_t *f, float kp)
{
    f->q[0] = 1.0f; f->q[1] = f->q[2] = f->q[3] = 0.0f;
    f->kp = kp;
    f->sig_a = 2.0f;
}

void tars_mahony_init_from_acc(tars_mahony_t *f, const float acc[3])
{
    float n = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    float u[3] = {0.0f, 0.0f, 1.0f};
    if (n > 1e-6f) { u[0] = acc[0] / n; u[1] = acc[1] / n; u[2] = acc[2] / n; }
    float th = -asinf(clampf_(u[0], -1.0f, 1.0f));
    float ph = atan2f(u[1], u[2]);
    float cp = cosf(th * 0.5f), sp = sinf(th * 0.5f);
    float cr = cosf(ph * 0.5f), sr = sinf(ph * 0.5f);
    f->q[0] = cr * cp;
    f->q[1] = sr * cp;
    f->q[2] = cr * sp;
    f->q[3] = -sr * sp;
}

void tars_mahony_update(tars_mahony_t *f, const float gyro[3], const float acc[3], float dt)
{
    const float qw = f->q[0], qx = f->q[1], qy = f->q[2], qz = f->q[3];
    const float vx = 2.0f * (qx * qz - qw * qy);
    const float vy = 2.0f * (qw * qx + qy * qz);
    const float vz = qw * qw - qx * qx - qy * qy + qz * qz;
    float gx = gyro[0], gy = gyro[1], gz = gyro[2];

    const float n = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    if (n > 1e-3f) {
        const float ax = acc[0] / n, ay = acc[1] / n, az = acc[2] / n;
        const float ex = ay * vz - az * vy;
        const float ey = az * vx - ax * vz;
        const float ez = ax * vy - ay * vx;
        const float d = (n - TARS_G) / f->sig_a;
        const float k = f->kp * expf(-0.5f * d * d);
        gx += k * ex; gy += k * ey; gz += k * ez;
    }
    const float h = 0.5f * dt;
    float nqw = qw + (-qx * gx - qy * gy - qz * gz) * h;
    float nqx = qx + (qw * gx + qy * gz - qz * gy) * h;
    float nqy = qy + (qw * gy - qx * gz + qz * gx) * h;
    float nqz = qz + (qw * gz + qx * gy - qy * gx) * h;
    const float inv = 1.0f / sqrtf(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz);
    f->q[0] = nqw * inv; f->q[1] = nqx * inv; f->q[2] = nqy * inv; f->q[3] = nqz * inv;
}

void tars_mahony_up_body(const tars_mahony_t *f, float up[3])
{
    const float qw = f->q[0], qx = f->q[1], qy = f->q[2], qz = f->q[3];
    up[0] = 2.0f * (qx * qz - qw * qy);
    up[1] = 2.0f * (qw * qx + qy * qz);
    up[2] = qw * qw - qx * qx - qy * qy + qz * qz;
}

/* ------------------------------------------------------------------ gyro zero-rate offset */
static float norm3_(const float v[3]) { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

int tars_imu_axes_from_refs(const float up[3], const float forward[3], float rotation[9])
{
    const float up_norm = norm3_(up);
    if (!(up_norm >= 5.0f)) return 0;

    float z[3], x[3];
    for (int i = 0; i < 3; ++i) z[i] = up[i] / up_norm;
    const float projection = forward[0] * z[0] + forward[1] * z[1] + forward[2] * z[2];
    for (int i = 0; i < 3; ++i) x[i] = forward[i] - projection * z[i];
    const float forward_norm = norm3_(x);
    if (!(forward_norm >= 1.5f)) return 0;

    for (int i = 0; i < 3; ++i) x[i] /= forward_norm;
    const float y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]};
    for (int i = 0; i < 3; ++i) {
        rotation[i] = x[i];
        rotation[3 + i] = y[i];
        rotation[6 + i] = z[i];
    }
    return 1;
}

void tars_gyro_bias_reset(tars_gyro_bias_t *b) { memset(b, 0, sizeof(*b)); }

static void gyro_bias_clear_window_(tars_gyro_bias_t *b)
{
    b->n = 0;
    for (int k = 0; k < 3; k++) b->sum[k] = b->sum2[k] = b->acc[k] = 0.0f;
}

int tars_gyro_bias_start(tars_gyro_bias_t *b, const float acc[3], const float gyro[3], float mean_acc_out[3])
{
    if (fabsf(norm3_(acc) - TARS_G) < 1.5f && norm3_(gyro) < 0.5f) {
        for (int k = 0; k < 3; k++) {
            b->acc[k] += acc[k];
            b->sum[k] += gyro[k];
            b->sum2[k] += gyro[k] * gyro[k];
        }
        if (++b->n >= 30) {
            float mg[3];
            int quiet = 1;
            for (int k = 0; k < 3; k++) {
                mg[k] = b->sum[k] / (float)b->n;
                const float var = b->sum2[k] / (float)b->n - mg[k] * mg[k];
                if (var > 0.05f * 0.05f) quiet = 0;              /* being handled: wait for a really still moment */
            }
            if (quiet) {
                for (int k = 0; k < 3; k++) {
                    mean_acc_out[k] = b->acc[k] / (float)b->n;
                    b->bias[k] = mg[k];
                }
                gyro_bias_clear_window_(b);
                b->static_n = 0;
                return 1;
            }
            gyro_bias_clear_window_(b);
        }
    } else {
        gyro_bias_clear_window_(b);
    }
    return 0;
}

void tars_gyro_bias_apply(tars_gyro_bias_t *b, const float acc[3], float gyro[3])
{
    float dev = 0.0f;
    for (int k = 0; k < 3; k++) dev = fmaxf(dev, fabsf(gyro[k] - b->bias[k]));
    if (fabsf(norm3_(acc) - TARS_G) < 0.4f && dev < 0.03f) {       /* static: |a| ~ g and the gyro within 1.7 deg/s of the estimate */
        if (b->static_n < 1000000) b->static_n++;
        if (b->static_n > 100) {
            for (int k = 0; k < 3; k++) b->bias[k] += 0.005f * (gyro[k] - b->bias[k]);
        }
    } else {
        b->static_n = 0;
    }
    for (int k = 0; k < 3; k++) gyro[k] -= b->bias[k];
}

/* ------------------------------------------------------------------ policy wrapper */
void tars_policy_defaults(tars_policy_state_t *s, int residual)
{
    memset(s, 0, sizeof(*s));
    s->hip_scale = 0.6f;
    s->lift_scale = 0.014f;
    s->res_hip = 0.25f;
    s->res_lift = 0.008f;
    s->gait_freq = 1.25f;
    s->amp_per_vx = 1000.0f;
    s->amp_per_wz = 30.0f;
    s->yaw_sign = -1.0f;
    s->residual = residual;
    s->act_slew = 0.0f;
}

void tars_policy_reset(tars_policy_state_t *s)
{
    s->phase = 0.0f;
    s->have_prev = 0;
    memset(s->a_prev, 0, sizeof(s->a_prev));
    memset(s->prev_frame, 0, sizeof(s->prev_frame));
}

static int moving_(const tars_policy_state_t *s)
{
    return fabsf(s->cmd[0]) > 1e-4f || fabsf(s->cmd[1]) > 1e-4f;
}

void tars_baseline_action(const tars_policy_state_t *s, float a[3])
{
    const float ax = clampf_(s->amp_per_vx * s->cmd[0], -35.0f, 35.0f);
    const float ay = clampf_(s->amp_per_wz * s->cmd[1], -25.0f, 25.0f);
    a[0] = a[1] = a[2] = 0.0f;
    if (fabsf(ax) < 0.5f && fabsf(ay) < 0.5f) return;

    const float s_ = sinf(TARS_2PI * s->phase), c_ = cosf(TARS_2PI * s->phase);
    float ql, qr, ratio;
    if (fabsf(ay) >= fabsf(ax)) {
        ratio = fabsf(ay) / 25.0f;
        ql = s->yaw_sign * ay * s_;
        qr = -s->yaw_sign * ay * s_;
    } else {
        ratio = fabsf(ax) / 35.0f;
        ql = qr = ax * s_;
    }
    const float lift = (9.0f + 5.0f * ratio) * c_ / 1000.0f;
    a[0] = ql * TARS_COUNT_RAD / s->hip_scale;
    a[1] = qr * TARS_COUNT_RAD / s->hip_scale;
    a[2] = lift / s->lift_scale;
}

static void make_frame_(const tars_policy_state_t *s, const float up[3], const float gyro[3],
                        const float q[3], float f[TARS_OBS_FRAME])
{
    f[0] = -up[0]; f[1] = -up[1]; f[2] = -up[2];
    f[3] = gyro[0] * 0.25f; f[4] = gyro[1] * 0.25f; f[5] = gyro[2] * 0.25f;
    f[6] = q[0] / s->hip_scale;
    f[7] = q[1] / s->hip_scale;
    f[8] = q[2] / s->lift_scale;
    f[9] = s->a_prev[0]; f[10] = s->a_prev[1]; f[11] = s->a_prev[2];
}

void tars_build_obs(const tars_policy_state_t *s, const float up[3], const float gyro[3],
                    const float q[3], float obs[TARS_OBS_DIM])
{
    float frame[TARS_OBS_FRAME];
    make_frame_(s, up, gyro, q, frame);
    const float mv = moving_(s) ? 1.0f : 0.0f;
    obs[0] = s->cmd[0] / 0.08f;
    obs[1] = s->cmd[1] / 0.8f;
    obs[2] = sinf(TARS_2PI * s->phase) * mv;
    obs[3] = cosf(TARS_2PI * s->phase) * mv;
    memcpy(&obs[4], frame, sizeof(frame));
    memcpy(&obs[4 + TARS_OBS_FRAME], s->have_prev ? s->prev_frame : frame, sizeof(frame));
}

void tars_stand_init(tars_stand_t *s, int residual, float act_slew)
{
    memset(s, 0, sizeof(*s));
    tars_policy_defaults(&s->pol, residual);
    s->pol.act_slew = act_slew;
    s->scale_pct = 100;
    tars_policy_reset(&s->pol);
}

void tars_stand_reset(tars_stand_t *s)
{
    tars_policy_reset(&s->pol);
    s->prev[0] = s->prev[1] = s->prev[2] = 0.0f;
}

void tars_stand_step(tars_stand_t *s, const tars_mlp_t *mlp, const float up[3], const float gyro[3], float dt, float q[2], float *z)
{
    float act[3], tgt[3];
    tars_policy_step(&s->pol, mlp, up, gyro, s->prev, dt, act, tgt);
    const float k = 0.01f * (float)s->scale_pct;
    q[0] = tgt[0] * k;
    q[1] = tgt[1] * k;
    *z = tgt[2] * k;
    s->prev[0] = roundf(q[0] / TARS_COUNT_RAD) * TARS_COUNT_RAD;     /* what the servo bus gets: whole counts (lift: 0.13 mm) */
    s->prev[1] = roundf(q[1] / TARS_COUNT_RAD) * TARS_COUNT_RAD;
    s->prev[2] = roundf(*z / 0.00013f) * 0.00013f;
}

void tars_policy_step(tars_policy_state_t *s, const tars_mlp_t *mlp,
                      const float up[3], const float gyro[3], const float q_meas[3], float dt,
                      float action_out[3], float target_out[3])
{
    float obs[TARS_OBS_DIM], a_pol[TARS_ACT_DIM], a[3];
    float frame[TARS_OBS_FRAME];

    tars_build_obs(s, up, gyro, q_meas, obs);
    make_frame_(s, up, gyro, q_meas, frame);
    tars_mlp_forward(mlp, obs, a_pol);

    if (s->residual) {
        float base[3];
        tars_baseline_action(s, base);
        a[0] = clampf_(base[0] + a_pol[0] * s->res_hip / s->hip_scale, -1.0f, 1.0f);
        a[1] = clampf_(base[1] + a_pol[1] * s->res_hip / s->hip_scale, -1.0f, 1.0f);
        a[2] = clampf_(base[2] + a_pol[2] * s->res_lift / s->lift_scale, -1.0f, 1.0f);
    } else {
        a[0] = clampf_(a_pol[0], -1.0f, 1.0f);
        a[1] = clampf_(a_pol[1], -1.0f, 1.0f);
        a[2] = clampf_(a_pol[2], -1.0f, 1.0f);
    }

    if (s->act_slew > 0.0f) {                         /* the simulator limits the applied action's change per step: the network cannot chatter at the Nyquist rate */
        for (int i = 0; i < 3; i++) a[i] = clampf_(a[i], s->a_prev[i] - s->act_slew, s->a_prev[i] + s->act_slew);
    }
    target_out[0] = a[0] * s->hip_scale;
    target_out[1] = a[1] * s->hip_scale;
    target_out[2] = a[2] * s->lift_scale;
    for (int i = 0; i < 3; i++) { action_out[i] = a[i]; s->a_prev[i] = a[i]; }
    memcpy(s->prev_frame, frame, sizeof(frame));
    s->have_prev = 1;

    if (moving_(s)) {
        s->phase += s->gait_freq * dt;
        s->phase -= floorf(s->phase);
    } else {
        s->phase = 0.0f;
    }
}
