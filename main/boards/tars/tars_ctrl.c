/* tars_ctrl.c - see tars_ctrl.h.  The sequencing mirrors rl/getup_seq.py and rl/walk_cma.py (validated in the simulator). */
#include "tars_ctrl.h"

#include <math.h>
#include <string.h>

#define TWO_PI 6.28318530718f

static float clampf_(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static float slew_(float cur, float tgt, float rate, float dt)
{
    float d = tgt - cur, m = rate * dt;
    return cur + clampf_(d, -m, m);
}

int tars_gait_in_range(const tars_gait_t *g)
{
    /* same order and bounds as NAMES / LO / HI of tars_rl/rl/walk_cma.py */
    static const float lo[15] = {-0.10f, -0.30f, -0.30f, -0.15f, -0.15f, -0.006f, -0.012f, -0.012f, -0.006f, -0.006f, 0.6f, -4.0f, -1.0f, -0.10f, -0.03f};
    static const float hi[15] = {0.10f, 0.30f, 0.30f, 0.15f, 0.15f, 0.006f, 0.012f, 0.012f, 0.006f, 0.006f, 2.0f, 4.0f, 1.0f, 0.10f, 0.03f};
    const float v[15] = {g->q0, g->a1, g->b1, g->a2, g->b2, g->z0, g->c1, g->d1, g->c2, g->d2, g->f, g->kp_h, g->kd_h, g->kp_z, g->kd_z};
    for (int i = 0; i < 15; i++) {
        const float m = 0.02f * (hi[i] - lo[i]);
        if (!(v[i] >= lo[i] - m && v[i] <= hi[i] + m)) return 0;      /* also rejects NaN */
    }
    return 1;
}

const char *tars_state_name(tars_state_t s)
{
    static const char *n[] = {"STAND", "WALK_F", "WALK_B", "TURN_L", "TURN_R", "FALLEN",
                              "GETUP1", "GETUP2A", "GETUP2B", "GETUP2C", "FAILED", "GETUP_NN", "MOTION"};
    return (unsigned)s < sizeof(n) / sizeof(n[0]) ? n[s] : "?";
}

void tars_ctrl_init(tars_ctrl_t *c)
{
    memset(c, 0, sizeof(*c));
    c->fall_up_thresh = 0.5f;
    c->fall_time = 0.3f;
    c->still_time = 0.6f;
    c->lift_up = 0.013f;
    c->hip_rate_down = 1.2f;
    c->max_attempts = 3;
    c->ramp_time = 1.0f;
    c->cmd_deadband = 1.0f;
    c->fb_kp_scale = c->fb_kd_scale = c->amp_scale = 1.0f;
    c->stand_up_min = 0.8f;
    c->speed_map = 1;
    c->speed_lo = 14.0f;
    c->speed_hi = 20.0f;
    c->speed_kmin = 0.70f;
    c->speed_kmax = 0.90f;
    c->eps_max = 0.30f;
    c->straight = 1;                 /* heading hold ON from power-up (tested on the real robot, REPORT section 19); `learn straight off` switches it off for experiments */
    c->hold_kp = 1.5f;               /* stroke fraction per rad of heading error (about 2.6 % per degree).  The real robot turns 2.5..3.5 times less per unit of stroke
                                      * difference than the calibrated simulator (6..23 deg/s per unit instead of 43), which would want 0.55: the loop closes in ~2..6 s */
    c->heading_max = 0.26f;          /* the heading error that is remembered is limited to 15 deg: a robot that is turned by hand does not keep "unwinding" */
    c->hold_kd = 0.0f;
    c->trim = 0.0f;
    c->k_cmd = 1.0f;
    c->fb_lpf_hz = 0.0f;
    c->getup_nn_mode = 0;            /* the portable core starts with the keyframe get-up; the firmware glue (tars_rl.cc) installs the network and its default mode (1) */
    c->nn_timeout = 7.0f;
    c->nn_hip_rate = 8.0f;
    c->nn_lift_rate = 0.25f;
    c->motion_wait_max = 3.0f;
    c->motion_start_up = 0.96f;
    c->motion_home_rate_hip = 2.0f;
    c->motion_home_rate_lift = 0.05f;
    c->state = TARS_S_STAND;
}

void tars_ctrl_reset(tars_ctrl_t *c)
{
    c->state = TARS_S_STAND;
    c->t_state = c->t_gait = c->gain = 0.0f;
    c->fall_t = c->still_t = c->stable_t = c->moved_t = 0.0f;
    c->gy_f = 0.0f;
    c->gy_ok = 0;
    c->stand_active = 0;
    c->k_cmd = 1.0f;
    c->heading = 0.0f;
    c->eps = 0.0f;
    c->nn_acc = 0.0f;
    c->nn_have_pf = 0;
    memset(c->nn_a, 0, sizeof(c->nn_a));
    memset(c->nn_q, 0, sizeof(c->nn_q));
    c->attempts = 0;
    c->hc = c->zc = 0.0f;
    c->getup_sign = 1;
    c->motion_req = c->motion = NULL;                 /* a show motion does not survive a reset (learned mode off / on) */
    c->motion_phase = c->motion_abort = 0;
    c->motion_t = c->motion_tilt_t = c->motion_leg_t = c->motion_wait = 0.0f;
    memset(c->motion_q, 0, sizeof(c->motion_q));
}

static void gait_eval_(const tars_gait_t *p, float t, float up0, float gyro_y, float kp_s, float kd_s, float amp_s, float *q, float *z)
{
    const float ph = TWO_PI * p->f * t;
    const float c1 = cosf(ph), s1 = sinf(ph), c2 = cosf(2.0f * ph), s2 = sinf(2.0f * ph);
    float qq = p->q0 + (amp_s * p->a1) * c1 + (amp_s * p->b1) * s1 + (amp_s * p->a2) * c2 + (amp_s * p->b2) * s2;   /* amp_s = 1: the searched gait, bit for bit */
    float zz = p->z0 + (amp_s * p->c1) * c1 + (amp_s * p->d1) * s1 + (amp_s * p->c2) * c2 + (amp_s * p->d2) * s2;
    const float theta = -asinf(clampf_(up0, -1.0f, 1.0f));   /* >0: torso pitched forward */
    qq += (kp_s * p->kp_h) * theta + (kd_s * p->kd_h) * gyro_y;
    zz += (kp_s * p->kp_z) * theta + (kd_s * p->kd_z) * gyro_y;
    *q = qq;
    *z = zz;
}

static float interp_(const float *v, int k, float tend, float t)
{
    if (t <= 0.0f) return clampf_(v[0], -1.0f, 1.0f);
    if (t >= tend) return clampf_(v[k - 1], -1.0f, 1.0f);
    const float pos = t / tend * (float)(k - 1);
    int i = (int)pos;
    if (i > k - 2) i = k - 2;
    const float fr = pos - (float)i;
    return clampf_(v[i], -1.0f, 1.0f) * (1.0f - fr) + clampf_(v[i + 1], -1.0f, 1.0f) * fr;
}

static tars_state_t decide_(const tars_ctrl_t *c, float vx, float vyaw)
{
    const float ax = fabsf(vx), ay = fabsf(vyaw);
    if (ax < c->cmd_deadband && ay < c->cmd_deadband) return TARS_S_STAND;
    if (ay >= ax) {
        if (vyaw > 0.0f) return c->gait_turn ? TARS_S_TURN_L : TARS_S_STAND;
        return (c->gait_turn_r || c->gait_turn) ? TARS_S_TURN_R : TARS_S_STAND;
    }
    if (vx > 0.0f) return c->gait_f ? TARS_S_WALK_F : TARS_S_STAND;
    return c->gait_b ? TARS_S_WALK_B : TARS_S_STAND;
}

static int is_walk_(tars_state_t s) { return s >= TARS_S_WALK_F && s <= TARS_S_TURN_R; }

/* amplitude scale asked for by the magnitude of the walking command (servo counts) */
static float speed_scale_(const tars_ctrl_t *c, tars_state_t s, float vx, float vyaw)
{
    if (!c->speed_map) return 1.0f;
    const float mag = (s == TARS_S_TURN_L || s == TARS_S_TURN_R) ? fabsf(vyaw) : fabsf(vx);
    const float span = c->speed_hi - c->speed_lo;
    const float u = span > 1e-3f ? (mag - c->speed_lo) / span : 1.0f;
    const float kmax = c->speed_kmax < c->speed_kmin ? c->speed_kmin : c->speed_kmax;
    return clampf_(c->speed_kmin + (kmax - c->speed_kmin) * u, c->speed_kmin, kmax);
}

/* position in a motion table (linear interpolation; q = {left hip [rad], right hip [rad], lift [m]}) */
static void motion_eval_(const tars_motion_t *m, float t, float q[3])
{
    float pos = t / m->dt;
    int i;
    if (pos < 0.0f) pos = 0.0f;
    if (pos >= (float)(m->n - 1)) { i = m->n - 2; pos = (float)(m->n - 1); }
    else i = (int)pos;
    const float f = pos - (float)i;
    q[0] = ((float)m->hl[i] * (1.0f - f) + (float)m->hl[i + 1] * f) * 1e-3f;
    q[1] = ((float)m->hr[i] * (1.0f - f) + (float)m->hr[i + 1] * f) * 1e-3f;
    q[2] = ((float)m->z[i] * (1.0f - f) + (float)m->z[i + 1] * f) * 1e-4f;
}

int tars_ctrl_motion_start(tars_ctrl_t *c, const tars_motion_t *m, float speed)
{
    if (!m || m->n < 2 || m->dt <= 0.0f || !m->hl || !m->hr || !m->z) return -1;
    if (c->state != TARS_S_STAND && !(c->state >= TARS_S_WALK_F && c->state <= TARS_S_TURN_R)) return -3;     /* falling, getting up, failed, or already performing */
    if (c->motion_req) return -3;
    c->motion_req_speed = clampf_(speed, 0.3f, 1.0f);
    c->motion_wait = 0.0f;
    c->motion_end = 0;
#if defined(__GNUC__)
    __sync_synchronize();                                /* the fields above must be visible to the other core before the pointer is */
#endif
    c->motion_req = m;                                   /* last: the control task looks at this one */
    return 0;
}

void tars_ctrl_motion_stop(tars_ctrl_t *c)
{
    if (c->motion_req) { c->motion_req = NULL; c->motion_end = 2; }
    if (c->state == TARS_S_MOTION) c->motion_abort = 1;
}

#define NN_HIP_RANGE  2.4f      /* the network's action 1.0 = 2.4 rad on the hips and 14 mm on the lift (the scales it was trained with) */
#define NN_LIFT_RANGE 0.014f

/* one observation frame of the get-up network (sim/tars_env.py _frame): -up, gyro/4, applied targets (hips / hip_range, lift / lift_range), previous action (= the same, not quantised) */
static void nn_frame_(const tars_ctrl_t *c, const float up[3], const float gyro[3], float frame[12])
{
    for (int i = 0; i < 3; i++) { frame[i] = -up[i]; frame[3 + i] = gyro[i] * 0.25f; }
    frame[6] = c->nn_q[0] / NN_HIP_RANGE;
    frame[7] = c->nn_q[1] / NN_HIP_RANGE;
    frame[8] = c->nn_q[2] / NN_LIFT_RANGE;
    for (int i = 0; i < 3; i++) frame[9 + i] = frame[6 + i];
}

void tars_ctrl_step(tars_ctrl_t *c, const float up[3], const float gyro[3], float dt,
                    float cmd_vx, float cmd_vyaw, float out_q[2], float *out_z)
{
    const float wnorm = sqrtf(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
    float zt = 0.0f;
    float ql = 0.0f, qr = 0.0f;
    int direct = 0;                        /* walking writes targets directly (as in the CMA rollouts) */

    c->t_state += dt;
    if (c->motion_req && c->state > TARS_S_TURN_R) { c->motion_req = NULL; c->motion_end = 5; }     /* a motion request does not outlive a fall / get-up / another motion */

    /* pitch rate for the walking feedback: optionally low-passed (the real robot's loop is slower than the simulated one) */
    float gy = gyro[1];
    if (c->fb_lpf_hz > 0.0f && dt > 0.0f) {
        if (!c->gy_ok) { c->gy_f = gyro[1]; c->gy_ok = 1; }
        c->gy_f += (dt / (dt + 1.0f / (TWO_PI * c->fb_lpf_hz))) * (gyro[1] - c->gy_f);
        gy = c->gy_f;
    } else {
        c->gy_ok = 0;
    }

    /* ------------------------------------------------ fall detection (all states except the get-up and the show motions, which have their own limits) */
    if (c->state != TARS_S_GETUP1 && c->state != TARS_S_GETUP_NN && c->state != TARS_S_FALLEN && c->state != TARS_S_FAILED && c->state != TARS_S_MOTION) {
        if (up[2] < c->fall_up_thresh) c->fall_t += dt; else c->fall_t = 0.0f;
        if (c->fall_t >= c->fall_time) {
            c->state = TARS_S_FALLEN;
            c->t_state = 0.0f;
            c->still_t = 0.0f;
            c->attempts = 0;
            c->gain = 0.0f;
        }
    }

    switch (c->state) {
    case TARS_S_STAND:
    case TARS_S_WALK_F:
    case TARS_S_WALK_B:
    case TARS_S_TURN_L:
    case TARS_S_TURN_R: {
        if (c->motion_req) {                       /* a requested show motion starts from a still, upright stand with no walking command */
            const tars_motion_t *m = c->motion_req;
            c->motion_wait += dt;
            if (c->state == TARS_S_STAND && up[2] > c->motion_start_up && wnorm < 1.0f && fabsf(cmd_vx) < c->cmd_deadband && fabsf(cmd_vyaw) < c->cmd_deadband) {
                c->motion = m;
                c->motion_req = NULL;
                c->motion_speed = c->motion_req_speed;
                c->motion_t = 0.0f;
                c->motion_phase = c->motion_abort = 0;
                c->motion_tilt_t = c->motion_leg_t = 0.0f;
                c->motion_end = 0;
                c->motion_q[0] = c->motion_q[1] = c->hc;
                c->motion_q[2] = c->zc;
                c->state = TARS_S_MOTION;
                c->t_state = 0.0f;
                c->gain = 0.0f;
                ql = qr = c->hc; zt = c->zc;          /* the first step of the motion keeps the pose of the stand */
                out_q[0] = ql; out_q[1] = qr; *out_z = zt;
                return;
            }
            if (c->motion_wait > c->motion_wait_max) { c->motion_req = NULL; c->motion_end = 5; }
        }
        const tars_state_t want = decide_(c, cmd_vx, cmd_vyaw);
        if (is_walk_(c->state)) {
            if (want == c->state) {
                c->gain = fminf(1.0f, c->gain + dt / c->ramp_time);
            } else {
                c->gain = fmaxf(0.0f, c->gain - dt / c->ramp_time);
                if (c->gain <= 0.0f) { c->state = is_walk_(want) ? want : TARS_S_STAND; c->t_gait = 0.0f; c->gain = 0.0f; }
            }
        } else if (is_walk_(want)) {
            c->state = want; c->t_gait = 0.0f; c->gain = 0.0f;
            c->k_cmd = speed_scale_(c, want, cmd_vx, cmd_vyaw);        /* start at the asked speed (no slew) */
        }
        if (is_walk_(c->state) && want == c->state) {
            const float kt = speed_scale_(c, c->state, cmd_vx, cmd_vyaw);
            c->k_cmd = slew_(c->k_cmd, kt, 0.6f, dt);                  /* a changed speed command takes effect smoothly (0.6 per s) */
        }
        if (is_walk_(c->state)) {
            const tars_gait_t *g = c->state == TARS_S_WALK_F ? c->gait_f : (c->state == TARS_S_WALK_B ? c->gait_b : c->gait_turn);
            if (c->state == TARS_S_TURN_R && c->gait_turn_r) g = c->gait_turn_r;   /* dedicated right-turn gait (pattern q,-q) */
            float q, z;
            gait_eval_(g, c->t_gait, up[0], gy, c->fb_kp_scale, c->fb_kd_scale, c->amp_scale * c->k_cmd, &q, &z);
            q = clampf_(q * c->gain, -0.6f, 0.6f);        /* same limits as the search (hip +-0.6 rad, lift +-14 mm) */
            z = clampf_(z * c->gain, -0.014f, 0.014f);
            if (c->state == TARS_S_TURN_L) { ql = q; qr = -q; }
            else if (c->state == TARS_S_TURN_R) { if (c->gait_turn_r) { ql = q; qr = -q; } else { ql = -q; qr = q; } }
            else {
                /* walking straight: trim + heading hold (hips with unequal strokes) */
                if (c->t_gait == 0.0f) c->heading = 0.0f;
                if (c->straight) {
                    c->heading = clampf_(c->heading + gyro[2] * dt * c->gain, -c->heading_max, c->heading_max);
                    /* + = "turn right (clockwise)" for both directions: a bigger left stroke turns a forward walker to the right and a backward walker to the left */
                    const float dir = c->state == TARS_S_WALK_F ? 1.0f : -1.0f;
                    float eps = dir * (c->trim + c->hold_kp * c->heading + c->hold_kd * gyro[2]);
                    eps = clampf_(eps, -c->eps_max, c->eps_max) * c->gain;
                    c->eps = eps;
                    ql = q * (1.0f + eps);
                    qr = q * (1.0f - eps);
                } else {
                    c->eps = 0.0f;
                    ql = qr = q;
                }
            }
            zt = z;
            c->t_gait += dt;
            direct = 1;
        } else if (c->stand_on && c->stand_step && up[2] > c->stand_up_min) {
            float qa[2] = {0.0f, 0.0f}, za = 0.0f;       /* stand with the learned push-recovery assist */
            if (!c->stand_active && c->stand_reset) c->stand_reset(c->stand_ctx);
            c->stand_step(c->stand_ctx, up, gyro, dt, qa, &za);
            ql = clampf_(qa[0], -0.6f, 0.6f);
            qr = clampf_(qa[1], -0.6f, 0.6f);
            zt = clampf_(za, -0.014f, 0.014f);
            c->stand_active = 1;
            direct = 1;
        } else {
            ql = qr = 0.0f; zt = 0.0f;       /* stand: slew to the nominal pose */
            c->stand_active = 0;
            c->hc = slew_(c->hc, 0.0f, 1.2f, dt);
            c->zc = slew_(c->zc, 0.0f, 0.05f, dt);
            ql = qr = c->hc; zt = c->zc;
            direct = 1;
        }
        if (direct) { c->hc = ql; c->zc = zt; }
        break;
    }

    case TARS_S_FALLEN: {
        /* relax to the nominal pose and wait until the robot is still */
        c->hc = slew_(c->hc, 0.0f, 3.0f, dt);
        c->zc = slew_(c->zc, 0.0f, 0.25f, dt);
        ql = qr = c->hc; zt = c->zc;
        if (wnorm < 0.3f) c->still_t += dt; else c->still_t = 0.0f;
        if (c->still_t >= c->still_time) {
            if (fabsf(up[1]) > 0.8f || up[2] > 0.5f) {         /* on its side / not actually fallen */
                c->state = up[2] > 0.5f ? TARS_S_STAND : TARS_S_FAILED;
            } else {
                if (c->attempts >= c->max_attempts) {
                    c->state = TARS_S_FAILED;
                } else {
                    c->getup_sign = up[0] > 0.0f ? 1 : -1;      /* front face up => lying on the back (re-evaluated for every attempt) */
                    const int use_nn = c->getup_nn_fn && (c->getup_nn_mode == 2 || (c->getup_nn_mode == 1 && (c->attempts % 2) == 0));
                    c->state = use_nn ? TARS_S_GETUP_NN : TARS_S_GETUP1;
                    if (use_nn) {                                /* the network starts from where the relaxed arms are */
                        c->nn_q[0] = c->nn_q[1] = c->hc;
                        c->nn_q[2] = c->zc;
                        c->nn_a[0] = c->nn_a[1] = c->hc / NN_HIP_RANGE;
                        c->nn_a[2] = c->zc / NN_LIFT_RANGE;
                        c->nn_acc = 0.1f;                        /* decide at once */
                        c->nn_have_pf = 0;
                    }
                    c->t_state = 0.0f;
                    c->stable_t = 0.0f;
                    c->attempts++;
                }
            }
        }
        break;
    }

    case TARS_S_GETUP1: {
        const tars_getup_t *g = c->getup;
        const float tt = c->t_state < g->tend ? c->t_state : g->tend;
        const float h = interp_(g->hip, g->k, g->tend, tt);
        const float z = interp_(g->lift, g->k, g->tend, tt);
        float ht = (float)c->getup_sign * h * g->hip_range;
        float zt_ = z * g->lift_range;
        if (up[2] > 0.97f && wnorm < 0.6f) c->stable_t += dt; else c->stable_t = 0.0f;
        if (c->stable_t >= 0.5f) {
            c->state = TARS_S_GETUP2A; c->t_state = 0.0f;
        } else if (c->t_state > g->tend + 1.0f) {                /* attempt over, not upright: relax, wait until still, re-classify, retry */
            ht = 0.0f; zt_ = 0.0f;
            if (wnorm < 0.2f) {
                c->state = TARS_S_FALLEN;                          /* attempts counter is kept */
                c->t_state = 0.0f;
                c->still_t = 0.0f;
                c->stable_t = 0.0f;
            }
        }
        c->hc = slew_(c->hc, ht, g->rate, dt);
        c->zc = slew_(c->zc, zt_, 0.25f, dt);
        ql = qr = c->hc; zt = c->zc;
        break;
    }

    case TARS_S_GETUP_NN: {
        /* the network acts every 0.1 s (as in the training); the commanded targets follow the held action at the servo speed limit */
        float frame[12];
        c->nn_acc += dt;
        if (!c->nn_have_pf && c->nn_acc >= 0.08f) {              /* the second frame of the observation is the one 20 ms before the decision */
            nn_frame_(c, up, gyro, c->nn_pf);
            c->nn_have_pf = 1;
        }
        if (c->nn_acc >= 0.1f) {
            float obs[28], act[3];
            nn_frame_(c, up, gyro, frame);
            obs[0] = fminf(c->t_state / 10.0f, 1.5f);              /* elapsed time in the slot of the walking command (unused in the get-up) */
            obs[1] = obs[2] = obs[3] = 0.0f;
            memcpy(&obs[4], frame, sizeof(frame));
            memcpy(&obs[16], c->nn_have_pf ? c->nn_pf : frame, sizeof(frame));
            c->getup_nn_fn(c->getup_nn_ctx, obs, act);
            for (int i = 0; i < 3; i++) c->nn_a[i] = clampf_(act[i], -1.0f, 1.0f);
            c->nn_acc -= 0.1f;
            c->nn_have_pf = 0;
        }
        c->nn_q[0] = slew_(c->nn_q[0], c->nn_a[0] * NN_HIP_RANGE, c->nn_hip_rate, dt);
        c->nn_q[1] = slew_(c->nn_q[1], c->nn_a[1] * NN_HIP_RANGE, c->nn_hip_rate, dt);
        c->nn_q[2] = slew_(c->nn_q[2], c->nn_a[2] * NN_LIFT_RANGE, c->nn_lift_rate, dt);
        ql = c->nn_q[0]; qr = c->nn_q[1]; zt = c->nn_q[2];
        c->hc = 0.5f * (ql + qr);                                /* the shared commands follow, so that STAND / FALLEN continue from here */
        c->zc = zt;
        /* upright, still and the arms at the nominal pose for a second = done (the same test as in the training) */
        if (up[2] > 0.97f && wnorm < 0.6f && fabsf(ql) < 0.35f && fabsf(qr) < 0.35f && fabsf(zt) < 0.012f) c->stable_t += dt; else c->stable_t = 0.0f;
        if (c->stable_t >= 1.0f) {
            c->state = TARS_S_STAND; c->t_state = 0.0f; c->gain = 0.0f; c->fall_t = 0.0f;
        } else if (c->t_state > c->nn_timeout) {                 /* not standing in time: relax, wait until still, retry (the next attempt may use the keyframes) */
            c->state = TARS_S_FALLEN; c->t_state = 0.0f; c->still_t = 0.0f; c->stable_t = 0.0f;
        }
        break;
    }

    case TARS_S_MOTION: {
        const tars_motion_t *m = c->motion;
        /* the torso tilting far beyond what the motion needs: the robot is falling - end it as a fall (FALLEN waits until it is still and gets up) */
        if (up[2] < cosf(m->tilt_limit_deg * 0.017453292f)) c->motion_tilt_t += dt; else c->motion_tilt_t = 0.0f;
        /* the legs stand on the floor while the lift is extended: then the torso pitch follows the hips.  A pitch far from the hips' angle means the whole robot is rocking over
         * its toes - end the motion as a fall before it goes further (real robot, normal runs: |pitch - hip| <= 14 deg; the full-speed sneeze that tipped it over: 63 deg) */
        if (c->motion_phase == 0 && c->motion_q[2] < -0.006f) {
            const float pitch = asinf(clampf_(-up[0], -1.0f, 1.0f));                 /* + = nose down */
            if (fabsf(pitch - 0.5f * (c->motion_q[0] + c->motion_q[1])) > 0.52f) c->motion_leg_t += dt; else c->motion_leg_t = 0.0f;     /* 30 deg */
        } else {
            c->motion_leg_t = 0.0f;
        }
        if (c->motion_tilt_t >= 0.2f || c->motion_leg_t >= 0.15f) {
            c->state = TARS_S_FALLEN; c->t_state = 0.0f; c->still_t = 0.0f; c->attempts = 0; c->gain = 0.0f; c->motion_end = c->motion_leg_t >= 0.15f ? 6 : 4; c->motion = NULL;
            c->hc = 0.5f * (c->motion_q[0] + c->motion_q[1]); c->zc = c->motion_q[2];
            ql = c->motion_q[0]; qr = c->motion_q[1]; zt = c->motion_q[2];
            break;
        }
        if (c->motion_phase == 0 && !c->motion_abort && (fabsf(cmd_vx) >= c->cmd_deadband || fabsf(cmd_vyaw) >= c->cmd_deadband)) {
            c->motion_abort = 1;
            c->motion_end = 3;                                   /* somebody wants the robot to walk */
        }
        if (c->motion_phase == 0) {
            if (c->motion_abort) {
                c->motion_phase = 1;                             /* cut short: from where it is, hips home first, then the lift */
                if (!c->motion_end) c->motion_end = 2;
            } else {
                const float tend = (float)(m->n - 1) * m->dt;
                c->motion_t += dt * c->motion_speed;
                if (c->motion_t >= tend) { c->motion_t = tend; c->motion_phase = 1; c->motion_end = 1; }
                motion_eval_(m, c->motion_t, c->motion_q);
            }
        }
        if (c->motion_phase == 1) {
            c->motion_q[0] = slew_(c->motion_q[0], 0.0f, c->motion_home_rate_hip, dt);
            c->motion_q[1] = slew_(c->motion_q[1], 0.0f, c->motion_home_rate_hip, dt);
            if (fabsf(c->motion_q[0]) < 0.01f && fabsf(c->motion_q[1]) < 0.01f) c->motion_phase = 2;
        } else if (c->motion_phase == 2) {
            c->motion_q[0] = slew_(c->motion_q[0], 0.0f, c->motion_home_rate_hip, dt);
            c->motion_q[1] = slew_(c->motion_q[1], 0.0f, c->motion_home_rate_hip, dt);
            c->motion_q[2] = slew_(c->motion_q[2], 0.0f, c->motion_home_rate_lift, dt);
            if (fabsf(c->motion_q[0]) < 0.003f && fabsf(c->motion_q[1]) < 0.003f && fabsf(c->motion_q[2]) < 0.0003f) {
                c->motion_q[0] = c->motion_q[1] = c->motion_q[2] = 0.0f;
                c->state = TARS_S_STAND; c->t_state = 0.0f; c->gain = 0.0f; c->fall_t = 0.0f;
                c->motion = NULL;
            }
        }
        ql = c->motion_q[0]; qr = c->motion_q[1]; zt = c->motion_q[2];
        c->hc = 0.5f * (ql + qr);                                /* the shared commands follow, so that STAND / FALLEN continue from here */
        c->zc = zt;
        break;
    }

    case TARS_S_GETUP2A:                                         /* retract the legs, hold the arms */
        c->zc = slew_(c->zc, c->lift_up, 0.05f, dt);
        ql = qr = c->hc; zt = c->zc;
        if (fabsf(c->zc - c->lift_up) < 0.0005f) { c->state = TARS_S_GETUP2B; c->t_state = 0.0f; }
        break;

    case TARS_S_GETUP2B:                                         /* arms down to vertical, feet stay clear */
        c->hc = slew_(c->hc, 0.0f, c->hip_rate_down, dt);
        c->zc = slew_(c->zc, c->lift_up, 0.05f, dt);
        ql = qr = c->hc; zt = c->zc;
        if (fabsf(c->hc) < 0.02f) { c->state = TARS_S_GETUP2C; c->t_state = 0.0f; }
        break;

    case TARS_S_GETUP2C:                                         /* extend the legs gently: four feet down */
        c->hc = slew_(c->hc, 0.0f, c->hip_rate_down, dt);
        c->zc = slew_(c->zc, 0.0f, 0.012f, dt);
        ql = qr = c->hc; zt = c->zc;
        if (fabsf(c->zc) < 0.0003f) { c->state = TARS_S_STAND; c->t_state = 0.0f; c->gain = 0.0f; c->fall_t = 0.0f; }
        break;

    case TARS_S_FAILED:
    default:
        c->hc = slew_(c->hc, 0.0f, 1.0f, dt);
        c->zc = slew_(c->zc, 0.0f, 0.05f, dt);
        ql = qr = c->hc; zt = c->zc;
        /* Given up.  When somebody handles the robot (picked up / moved: rate > 0.8 rad/s) and it is still again, look at it afresh:
         * FALLEN decides at once (upright -> STAND, on its back / front -> get-up with a fresh attempt counter, on its side -> FAILED). */
        if (wnorm > 0.8f) { c->moved_t += dt; c->still_t = 0.0f; }
        else if (wnorm < 0.3f) c->still_t += dt;
        else c->still_t = 0.0f;
        if (c->moved_t >= 0.2f && c->still_t >= c->still_time) {
            c->moved_t = 0.0f;
            c->attempts = 0;
            c->fall_t = 0.0f;
            c->state = TARS_S_FALLEN;
            c->t_state = 0.0f;
        }
        break;
    }
    out_q[0] = ql;
    out_q[1] = qr;
    *out_z = zt;
}
