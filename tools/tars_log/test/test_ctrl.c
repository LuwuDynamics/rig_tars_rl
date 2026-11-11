/*
 * Host test of the learned supervisor (tars_ctrl.c, the very file that runs in the firmware): scripted "up" / gyro inputs, no physics.
 * Checks the fall -> get-up -> FAILED path and the recovery from FAILED when somebody picks the robot up and puts it down again.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tars_ctrl.h"
#include "tars_gait_params.h"
#include "tars_gait_params_v5.h"
#include "tars_gait_params_v7.h"
#include "tars_gait_params_v8.h"
#include "tars_gait_params_v9.h"
#include "tars_motions.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        g_checks++;                                                   \
        if (!(cond)) {                                                \
            g_fail++;                                                 \
            printf("  FAIL line %d: %s  ", __LINE__, #cond);          \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0)

static tars_ctrl_t C;
static double g_t = 0.0;

static void setup(void)
{
    tars_ctrl_init(&C);
    C.gait_f = &tars_gait_fwd;
    C.gait_b = TARS_GAIT_BACK_PTR;
    C.gait_turn = TARS_GAIT_TURN_PTR;
    C.gait_turn_r = TARS_GAIT_TURNR_PTR;
    C.getup = &tars_getup;
    tars_ctrl_reset(&C);
    g_t = 0.0;
}

/* run the supervisor for `sec` seconds at 50 Hz with constant inputs; returns the state at the end */
static tars_state_t run(const float up[3], float wrate, double sec)
{
    const float gyro[3] = {0.0f, wrate, 0.0f};
    float q[2], z;
    for (int i = 0; i < (int)(sec * 50.0 + 0.5); i++) {
        tars_ctrl_step(&C, up, gyro, 0.02f, 0.0f, 0.0f, q, &z);
        g_t += 0.02;
    }
    return C.state;
}

/* walk forward (command +20) for `sec` s with constant inputs; returns the left hip target [rad] of the last step */
static float walk_q(const float up[3], float wrate, double sec)
{
    const float gyro[3] = {0.0f, wrate, 0.0f};
    float q[2] = {0, 0}, z;
    for (int i = 0; i < (int)(sec * 50.0 + 0.5); i++) {
        tars_ctrl_step(&C, up, gyro, 0.02f, 20.0f, 0.0f, q, &z);
        g_t += 0.02;
    }
    return q[0];
}

/* dummy get-up network: constant action, remembers the last observation */
static int g_nn_calls = 0;
static float g_nn_obs[28], g_nn_first[28], g_nn_out[3] = {0.5f, 0.5f, 0.0f};
static void nn_dummy(void *ctx, const float obs[28], float act[3])
{
    (void)ctx;
    if (g_nn_calls == 0) memcpy(g_nn_first, obs, sizeof g_nn_first);
    g_nn_calls++;
    memcpy(g_nn_obs, obs, sizeof g_nn_obs);
    for (int i = 0; i < 3; i++) act[i] = g_nn_out[i];
}

/* dummy standing assist: constant targets, counts how often it is reset / stepped */
static int g_as_reset = 0, g_as_step = 0;
static void as_reset(void *ctx) { (void)ctx; g_as_reset++; }
static void as_step(void *ctx, const float up[3], const float gyro[3], float dt, float q[2], float *z)
{
    (void)ctx; (void)up; (void)gyro; (void)dt;
    g_as_step++;
    q[0] = 0.1f; q[1] = -0.2f; *z = 0.003f;
}

/* one supervisor step during a show motion: the attitude fed is that of a robot whose torso follows its hips (pitch = the hip angle that was commanded last), which is what the real
 * robot reports (|pitch - hip| <= 6 deg in normal runs); `extra` tilts it further (to test the guards) */
static void fstep_extra(float extra, float dt, float vx, float vyaw, float q[2], float *z)
{
    const float zero3[3] = {0, 0, 0};
    const float p = 0.5f * (q[0] + q[1]) + extra;
    const float up[3] = {-sinf(p), 0.0f, cosf(p)};
    tars_ctrl_step(&C, up, zero3, dt, vx, vyaw, q, z);
}
static void fstep(float dt, float vx, float vyaw, float q[2], float *z) { fstep_extra(0.0f, dt, vx, vyaw, q, z); }

int main(void)
{
    const float upright[3] = {0.0f, 0.0f, 1.0f};
    const float on_back[3] = {1.0f, 0.0f, 0.0f};          /* front face up */
    const float on_front[3] = {-1.0f, 0.0f, 0.0f};
    const float on_side[3] = {0.0f, 1.0f, 0.0f};

    setup();
    CHECK(run(upright, 0.0f, 2.0) == TARS_S_STAND, "upright stays STAND");

    /* lying on its back and never getting up: FALLEN -> GETUP1 (3 attempts, 8 s each) -> FAILED */
    CHECK(run(on_back, 0.0f, 1.2) == TARS_S_GETUP1, "state %d", C.state);
    CHECK(C.getup_sign == 1, "back-lying -> sign +1");
    CHECK(run(on_back, 0.0f, 40.0) == TARS_S_FAILED, "gave up after 3 attempts, state %d attempts %d", C.state, C.attempts);
    CHECK(C.attempts == C.max_attempts, "attempts %d", C.attempts);
    CHECK(run(on_back, 0.0f, 30.0) == TARS_S_FAILED, "stays FAILED while nobody touches it");

    /* somebody picks it up (rate 2 rad/s for 1 s) and puts it down on its back, still: a fresh get-up (attempt counter 1) */
    CHECK(run(on_back, 2.0f, 1.0) == TARS_S_FAILED, "being handled: still FAILED");
    CHECK(run(on_back, 0.0f, 1.0) == TARS_S_GETUP1, "put down on its back: get-up again, state %d", C.state);
    CHECK(C.attempts == 1, "fresh attempt counter %d", C.attempts);
    CHECK(C.getup_sign == 1, "sign +1");

    /* ... and when it is put down on its front the get-up mirrors */
    CHECK(run(on_back, 0.0f, 40.0) == TARS_S_FAILED, "fails again");
    CHECK(run(on_front, 2.0f, 1.0) == TARS_S_FAILED, "handled");
    CHECK(run(on_front, 0.0f, 1.0) == TARS_S_GETUP1 && C.getup_sign == -1, "front-lying -> sign -1, state %d sign %d", C.state, C.getup_sign);

    /* picked up and set on its feet: STAND at once */
    CHECK(run(on_front, 0.0f, 40.0) == TARS_S_FAILED, "fails once more");
    CHECK(run(upright, 2.0f, 1.0) == TARS_S_FAILED, "handled again");
    CHECK(run(upright, 0.0f, 1.0) == TARS_S_STAND, "set upright: STAND, state %d", C.state);

    /* lying on its side: gives up (cannot get up from there) and recovers when handled onto its back / feet */
    setup();
    CHECK(run(on_side, 0.0f, 3.0) == TARS_S_FAILED, "on its side: FAILED, state %d", C.state);
    CHECK(run(on_side, 0.0f, 20.0) == TARS_S_FAILED, "no retry while nobody touches it");
    CHECK(run(on_side, 2.0f, 0.5) == TARS_S_FAILED, "handled");
    CHECK(run(on_back, 0.0f, 1.5) == TARS_S_GETUP1, "put onto its back: get-up, state %d", C.state);

    /* tiny disturbances (rate < 0.8 rad/s) do not release FAILED */
    setup();
    run(on_back, 0.0f, 45.0);
    CHECK(C.state == TARS_S_FAILED, "FAILED %d", C.state);
    CHECK(run(on_back, 0.5f, 3.0) == TARS_S_FAILED, "a gentle vibration does not count as handling");
    CHECK(run(on_back, 0.0f, 3.0) == TARS_S_FAILED, "and still nothing afterwards");

    /* get-up works when the robot comes upright during GETUP1: GETUP2A/B/C -> STAND */
    setup();
    run(on_back, 0.0f, 3.0);
    CHECK(C.state == TARS_S_GETUP1, "state %d", C.state);
    run(upright, 0.0f, 1.0);
    CHECK(C.state == TARS_S_GETUP2A || C.state == TARS_S_GETUP2B || C.state == TARS_S_GETUP2C, "upright during get-up: state %d", C.state);
    CHECK(run(upright, 0.0f, 8.0) == TARS_S_STAND, "back to STAND, state %d", C.state);

    /* runtime tuning: defaults reproduce the searched gait; kd scale 0 removes the pitch-rate feedback, the low-pass slows it, amp scales the pattern */
    {
        const float upr[3] = {0.0f, 0.0f, 1.0f};
        const float tilt[3] = {-0.2f, 0.0f, 0.98f};                    /* leaning forward ~11.5 deg */
        const double sec = 5.0 + 1.0 / 3.0;                              /* the same gait phase for every run */
        setup();
        CHECK(C.fb_kp_scale == 1.0f && C.fb_kd_scale == 1.0f && C.fb_lpf_hz == 0.0f && C.amp_scale == 1.0f, "defaults");
        const float q_ref = walk_q(upr, 0.0f, sec);
        setup();
        const float q_rate = walk_q(upr, 3.0f, sec);                     /* 3 rad/s pitch rate */
        CHECK(fabsf(q_rate - q_ref) > 0.05f, "pitch-rate feedback moves the hip: %f vs %f", q_rate, q_ref);
        setup();
        C.fb_kd_scale = 0.0f;
        CHECK(fabsf(walk_q(upr, 3.0f, sec) - q_ref) < 1e-6f, "kd scale 0: the pitch rate no longer matters");
        setup();
        C.fb_kp_scale = 0.0f;
        const float q_nokp = walk_q(tilt, 0.0f, sec);
        setup();
        C.fb_kp_scale = 1.0f;
        const float q_kp = walk_q(tilt, 0.0f, sec);
        CHECK(fabsf(q_kp - q_nokp) > 0.05f, "pitch feedback moves the hip: %f vs %f", q_kp, q_nokp);
        setup();
        C.fb_kp_scale = 0.0f;
        CHECK(fabsf(walk_q(upr, 0.0f, sec) - q_ref) < 1e-6f, "kp scale 0 on an upright robot changes nothing");
        /* low-pass: a rate that flips sign every step (25 Hz) is almost removed, the same rate held constant gets through */
        setup();
        C.fb_lpf_hz = 3.0f;
        float q[2], z, last_alt = 0.0f;
        for (int i = 0; i < 270; i++) {
            const float gyro[3] = {0.0f, (i & 1) ? 3.0f : -3.0f, 0.0f};
            tars_ctrl_step(&C, upr, gyro, 0.02f, 20.0f, 0.0f, q, &z);
            last_alt = C.gy_f;
        }
        CHECK(fabsf(last_alt) < 0.6f && fabsf(last_alt) > 0.3f, "25 Hz (Nyquist) rate of amplitude 3 through the 3 Hz low-pass: %f (theory 0.476)", last_alt);
        setup();
        C.fb_lpf_hz = 3.0f;
        for (int i = 0; i < 270; i++) {
            const float gyro[3] = {0.0f, 3.0f, 0.0f};
            tars_ctrl_step(&C, upr, gyro, 0.02f, 20.0f, 0.0f, q, &z);
        }
        CHECK(fabsf(C.gy_f - 3.0f) < 0.05f, "constant rate passes: %f", C.gy_f);
        /* amplitude: 50% -> the periodic part is halved (q0 stays) */
        setup();
        C.fb_kp_scale = C.fb_kd_scale = 0.0f;
        float lo = 1e9f, hi = -1e9f, lo5 = 1e9f, hi5 = -1e9f;
        for (int pass = 0; pass < 2; pass++) {
            setup();
            C.fb_kp_scale = C.fb_kd_scale = 0.0f;
            C.amp_scale = pass ? 0.5f : 1.0f;
            const float gyro[3] = {0, 0, 0};
            for (int i = 0; i < 500; i++) {
                tars_ctrl_step(&C, upr, gyro, 0.02f, 20.0f, 0.0f, q, &z);
                if (i > 100) {
                    if (pass) { if (q[0] < lo5) lo5 = q[0]; if (q[0] > hi5) hi5 = q[0]; } else { if (q[0] < lo) lo = q[0]; if (q[0] > hi) hi = q[0]; }
                }
            }
        }
        CHECK(hi5 - lo5 > 0.3f * (hi - lo) && hi5 - lo5 < 0.7f * (hi - lo), "amp 50%%: swing %f vs %f", hi5 - lo5, hi - lo);
    }

    /* speed command: the magnitude of the walking command scales the gait amplitude (70 % at <= 14 counts ... 90 % at >= 20), the sign still picks the direction */
    {
        const float upr[3] = {0.0f, 0.0f, 1.0f}, zero3[3] = {0, 0, 0};
        float q[2], z, swing[3];
        const float cmds[3] = {10.0f, 17.0f, 20.0f};
        for (int k = 0; k < 3; k++) {
            setup();
            C.fb_kp_scale = C.fb_kd_scale = 0.0f;                /* pure pattern: the amplitude is what we look at */
            float lo = 1e9f, hi = -1e9f;
            for (int i = 0; i < 500; i++) {
                tars_ctrl_step(&C, upr, zero3, 0.02f, cmds[k], 0.0f, q, &z);
                if (i > 150) { if (q[0] < lo) lo = q[0]; if (q[0] > hi) hi = q[0]; }
            }
            swing[k] = hi - lo;
        }
        CHECK(swing[0] > 0.72f * swing[2] && swing[0] < 0.84f * swing[2], "speed 10 counts -> 70/90 of the swing of 20 counts: %f vs %f", swing[0], swing[2]);
        CHECK(swing[1] > swing[0] * 1.05f && swing[1] < swing[2] * 0.98f, "speed 17 counts in between: %f %f %f", swing[0], swing[1], swing[2]);
        setup();
        C.speed_map = 0;
        C.fb_kp_scale = C.fb_kd_scale = 0.0f;
        float lo = 1e9f, hi = -1e9f;
        for (int i = 0; i < 500; i++) {
            tars_ctrl_step(&C, upr, zero3, 0.02f, 10.0f, 0.0f, q, &z);
            if (i > 150) { if (q[0] < lo) lo = q[0]; if (q[0] > hi) hi = q[0]; }
        }
        CHECK(fabsf((hi - lo) - swing[2] / 0.9f) < 0.02f * swing[2], "speed map off: the full swing at any command (%f vs %f / 0.9)", hi - lo, swing[2]);
        setup();                                              /* backward and the sign */
        for (int i = 0; i < 100; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, -10.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_WALK_B && fabsf(C.k_cmd - 0.7f) < 0.02f, "-10 counts: WALK_B at 70 %% (state %d, k %f)", C.state, C.k_cmd);
    }

    /* walking straight: unequal hip strokes (trim / gyro heading hold) of the forward and backward walks only */
    {
        const float upr[3] = {0.0f, 0.0f, 1.0f}, zero3[3] = {0, 0, 0};
        float q[2], z;
        /* the defaults: on, kp 1.5, no trim */
        setup();
        CHECK(C.straight == 1 && fabsf(C.hold_kp - 1.5f) < 1e-6f && C.hold_kd == 0.0f && C.trim == 0.0f && C.eps_max == 0.3f, "defaults: on, kp %f kd %f trim %f", C.hold_kp, C.hold_kd, C.trim);
        /* off: both hips get exactly the same stroke, forward and backward, whatever the gyro says */
        for (int back = 0; back < 2; back++) {
            setup();
            C.straight = 0;
            float worst = 0.0f;
            for (int i = 0; i < 400; i++) {
                const float yaw_rate[3] = {0.0f, 0.0f, 0.3f};
                tars_ctrl_step(&C, upr, yaw_rate, 0.02f, back ? -20.0f : 20.0f, 0.0f, q, &z);
                if (fabsf(q[0] - q[1]) > worst) worst = fabsf(q[0] - q[1]);
            }
            CHECK(C.state == (back ? TARS_S_WALK_B : TARS_S_WALK_F) && worst == 0.0f && C.eps == 0.0f, "straight off: equal strokes (diff %g, eps %g)", worst, C.eps);
        }
        /* trim: + = turn right in both directions: forward the LEFT stroke gets bigger, backward the RIGHT one */
        for (int back = 0; back < 2; back++) {
            setup();
            C.straight = 1;
            C.hold_kp = C.hold_kd = 0.0f;
            C.trim = 0.1f;
            float best = 0.0f, ratio = 0.0f;
            for (int i = 0; i < 500; i++) {
                tars_ctrl_step(&C, upr, zero3, 0.02f, back ? -20.0f : 20.0f, 0.0f, q, &z);
                if (i > 150 && fabsf(q[0] + q[1]) > best) { best = fabsf(q[0] + q[1]); ratio = (q[0] - q[1]) / (q[0] + q[1]); }
            }
            CHECK(fabsf(ratio - (back ? -0.1f : 0.1f)) < 0.003f, "trim 0.1 %s: (left - right) / (left + right) = %f", back ? "backward" : "forward", ratio);
        }
        /* turns and standing are not touched */
        setup();
        C.straight = 1;
        C.trim = 0.2f;
        for (int i = 0; i < 400; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 20.0f, q, &z);
        CHECK(C.state == TARS_S_TURN_L && fabsf(q[0] + q[1]) < 1e-5f && C.eps == 0.0f, "turning: untouched (state %d, eps %g)", C.state, C.eps);
        /* closed loop with a toy yaw plant: yaw rate = natural drift + K * eps.  K as measured on the real robot (forward -17 deg/s per unit stroke difference,
         * backward +13; the calibrated simulator has -43 / +33) and a drift of 1.1 / 1.4 deg/s to the right (the real robot drifts to the right) */
        for (int back = 0; back < 2; back++) {
            const float K = back ? 0.23f : -0.30f, drift = back ? -0.025f : -0.02f;
            double psi_on = 0.0, psi_off = 0.0, worst_on = 0.0;
            for (int on = 0; on < 2; on++) {
                setup();
                C.straight = on;
                double psi = 0.0;
                for (int i = 0; i < 2000; i++) {                   /* 40 s */
                    const float yr = drift + K * C.eps;
                    const float g3[3] = {0.0f, 0.0f, yr};
                    tars_ctrl_step(&C, upr, g3, 0.02f, back ? -20.0f : 20.0f, 0.0f, q, &z);
                    psi += (double)yr * 0.02;
                    if (on && fabs(psi) > worst_on) worst_on = fabs(psi);
                }
                if (on) psi_on = psi; else psi_off = psi;
            }
            CHECK(fabs(psi_off - drift * 40.0) < 0.1, "%s without the hold the heading runs away: %f rad", back ? "backward" : "forward", psi_off);
            CHECK(fabs(psi_on) < 0.12 && worst_on < 0.2, "%s with the hold the heading stays within 0.12 rad: end %f, worst %f (open loop %f)", back ? "backward" : "forward", psi_on, worst_on, psi_off);
        }
        /* the heading restarts with every walk, and is limited */
        setup();
        C.straight = 1;
        C.eps_max = 0.3f;
        for (int i = 0; i < 300; i++) {
            const float g3[3] = {0.0f, 0.0f, 1.0f};               /* turning left at 57 deg/s all the time */
            tars_ctrl_step(&C, upr, g3, 0.02f, 20.0f, 0.0f, q, &z);
        }
        CHECK(fabsf(C.heading - C.heading_max) < 1e-4f && fabsf(C.eps) <= 0.3001f && C.eps > 0.29f, "a runaway heading is limited (heading_max %f): heading %f eps %f", C.heading_max, C.heading, C.eps);
        for (int i = 0; i < 300; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 0.0f, q, &z);       /* stop ... */
        for (int i = 0; i < 100; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 20.0f, 0.0f, q, &z);      /* ... and start again */
        CHECK(fabsf(C.heading) < 0.05f, "a new walk starts with heading 0: %f", C.heading);
    }

    /* learned get-up (neural network) state, with a dummy network: observation layout, timing, speed limit, success, timeout, attempt order */
    {
        const float back[3] = {1.0f, 0.0f, 0.0f}, upr[3] = {0.0f, 0.0f, 1.0f}, zero3[3] = {0, 0, 0};
        float q[2], z;
        /* mode 0 (default) or no network: the keyframes, as before */
        setup();
        C.getup_nn_fn = nn_dummy;
        CHECK(C.getup_nn_mode == 0, "default get-up mode is the keyframes");
        CHECK(run(back, 0.0f, 1.5) == TARS_S_GETUP1 && g_nn_calls == 0, "mode 0: keyframe get-up, network not called (state %d)", C.state);
        setup();
        C.getup_nn_mode = 2;                                  /* mode 2 without a network compiled in: keyframes */
        CHECK(run(back, 0.0f, 1.5) == TARS_S_GETUP1, "no network: keyframes (state %d)", C.state);
        /* mode 2: the network */
        setup();
        C.getup_nn_mode = 2;
        C.getup_nn_fn = nn_dummy;
        g_nn_calls = 0;
        g_nn_out[0] = g_nn_out[1] = 0.5f; g_nn_out[2] = 0.0f;
        int started = -1;
        for (int i = 0; i < 100; i++) {
            tars_ctrl_step(&C, back, zero3, 0.02f, 0.0f, 0.0f, q, &z);
            if (C.state == TARS_S_GETUP_NN && started < 0) started = i;
        }
        CHECK(started >= 40 && started <= 70, "the network starts after the fall detection (0.3 s) and the stillness wait (0.6 s): step %d", started);
        CHECK(g_nn_calls >= 1 && fabsf(g_nn_first[0]) < 0.01f && g_nn_first[1] == 0.0f && g_nn_first[2] == 0.0f && g_nn_first[3] == 0.0f, "first observation: time 0, unused slots 0 (%f)", g_nn_first[0]);
        CHECK(fabsf(g_nn_first[4] + 1.0f) < 1e-6f && g_nn_first[5] == 0.0f && g_nn_first[6] == 0.0f, "observation frame starts with -up (%f %f %f)", g_nn_first[4], g_nn_first[5], g_nn_first[6]);
        CHECK(memcmp(&g_nn_first[4], &g_nn_first[16], 12 * sizeof(float)) == 0, "the first previous frame is the current frame");
        CHECK(C.attempts == 1, "one attempt used (the fall detection must not restart it): %d", C.attempts);
        for (int i = 0; i < 50; i++) tars_ctrl_step(&C, back, zero3, 0.02f, 0.0f, 0.0f, q, &z);       /* 1 s of network */
        CHECK(C.state == TARS_S_GETUP_NN && g_nn_calls >= 19 && g_nn_calls <= 22, "a decision every 0.1 s: %d calls in about 2 s of network time", g_nn_calls);
        CHECK(fabsf(q[0] - 1.2f) < 0.01f && fabsf(q[1] - 1.2f) < 0.01f && fabsf(z) < 1e-5f, "hips went to 0.5 x 2.4 rad: %f %f lift %f", q[0], q[1], z);
        CHECK(fabsf(g_nn_obs[0] - (C.t_state / 10.0f)) < 0.012f && fabsf(g_nn_obs[10] - 0.5f) < 1e-3f && fabsf(g_nn_obs[11] - 0.5f) < 1e-3f, "observation: elapsed time %f, hips %f %f", g_nn_obs[0], g_nn_obs[10], g_nn_obs[11]);
        CHECK(fabsf(g_nn_obs[22] - 0.5f) < 1e-3f, "previous frame has the hips too (%f)", g_nn_obs[22]);
        /* the speed limit: 8 rad/s on the hips */
        setup();
        C.getup_nn_mode = 2; C.getup_nn_fn = nn_dummy; g_nn_out[0] = g_nn_out[1] = 1.0f;
        float prev = 0.0f, worst = 0.0f;
        for (int i = 0; i < 200; i++) {
            tars_ctrl_step(&C, back, zero3, 0.002f, 0.0f, 0.0f, q, &z);
            if (C.state == TARS_S_GETUP_NN) { if (fabsf(q[0] - prev) > worst) worst = fabsf(q[0] - prev); prev = q[0]; }
        }
        for (int i = 0; i < 500; i++) {
            tars_ctrl_step(&C, back, zero3, 0.002f, 0.0f, 0.0f, q, &z);
            if (fabsf(q[0] - prev) > worst) worst = fabsf(q[0] - prev);
            prev = q[0];
        }
        CHECK(worst > 0.0155f && worst <= 0.01601f, "hips move at most 8 rad/s (0.016 rad per 2 ms step): worst step %f", worst);
        /* timeout and attempts: the network keeps failing (the robot stays on its back) */
        setup();
        C.getup_nn_mode = 2; C.getup_nn_fn = nn_dummy; g_nn_out[0] = g_nn_out[1] = 0.5f;
        CHECK(run(back, 0.0f, 5.0) == TARS_S_GETUP_NN && C.attempts == 1, "5 s into the network attempt (attempts %d)", C.attempts);
        CHECK(run(back, 0.0f, 3.5) == TARS_S_FALLEN && C.attempts == 1, "timeout after 7 s: relaxed and waiting (state %d, attempts %d)", C.state, C.attempts);
        CHECK(run(back, 0.0f, 3.0) == TARS_S_GETUP_NN && C.attempts == 2, "second attempt is the network again in mode 2 (attempts %d)", C.attempts);
        CHECK(run(back, 0.0f, 40.0) == TARS_S_FAILED && C.attempts == 3, "three attempts, then FAILED (attempts %d)", C.attempts);
        /* mode 1: network, keyframes, network */
        setup();
        C.getup_nn_mode = 1; C.getup_nn_fn = nn_dummy;
        CHECK(run(back, 0.0f, 2.0) == TARS_S_GETUP_NN, "mode 1: the first attempt is the network (state %d)", C.state);
        run(back, 0.0f, 6.0);                                  /* timeout */
        CHECK(run(back, 0.0f, 3.0) == TARS_S_GETUP1 && C.attempts == 2, "mode 1: the second attempt is the keyframe get-up (state %d, attempts %d)", C.state, C.attempts);
        /* success: upright, still, arms nominal for a second -> STAND */
        setup();
        C.getup_nn_mode = 2; C.getup_nn_fn = nn_dummy; g_nn_out[0] = g_nn_out[1] = g_nn_out[2] = 0.0f;
        CHECK(run(back, 0.0f, 2.0) == TARS_S_GETUP_NN, "network running (state %d)", C.state);
        CHECK(run(upr, 0.0f, 0.6) == TARS_S_GETUP_NN, "upright for 0.6 s is not enough (state %d)", C.state);
        CHECK(run(upr, 0.0f, 0.8) == TARS_S_STAND, "upright, still and nominal for a second: STAND (state %d)", C.state);
        /* upright but the arms away from the nominal pose: not done */
        setup();
        C.getup_nn_mode = 2; C.getup_nn_fn = nn_dummy; g_nn_out[0] = g_nn_out[1] = 0.5f;
        run(back, 0.0f, 2.0);
        CHECK(run(upr, 0.0f, 3.0) == TARS_S_GETUP_NN, "arms at 1.2 rad: still not done (state %d)", C.state);
        g_nn_out[0] = g_nn_out[1] = 0.5f;
    }

    /* range check of gaits that arrive over the serial line: every compiled gait is inside, broken ones are not */
    {
        CHECK(tars_gait_in_range(&tars_gait_fwd) && tars_gait_in_range(&tars_gait_fwd_v5) && tars_gait_in_range(&tars_gait_fwd_v5_fast) && tars_gait_in_range(&tars_gait_fwd_v7),
              "compiled forward gaits are inside the search ranges");
        CHECK(tars_gait_in_range(&tars_gait_back_v5) && tars_gait_in_range(&tars_gait_back_v7) && tars_gait_in_range(&tars_gait_turn_v7) && tars_gait_in_range(&tars_gait_turn_r_v7),
              "compiled back / turn gaits are inside");
        CHECK(tars_gait_in_range(&tars_gait_fwd_v8) && tars_gait_in_range(&tars_gait_back_v8) && tars_gait_in_range(&tars_gait_turn_v8),
              "v8 gaits are inside the search ranges (the right turn is the mirror of the left one: no dedicated gait)");
        CHECK(tars_gait_in_range(&tars_gait_fwd_v9) && tars_gait_in_range(&tars_gait_back_v9) && tars_gait_in_range(&tars_gait_turn_v9), "v9 gaits are inside the search ranges");
        tars_gait_t g = tars_gait_fwd_v7;
        g.f = 0.1f;                                 CHECK(!tars_gait_in_range(&g), "f 0.1 Hz rejected");
        g = tars_gait_fwd_v7; g.f = 3.5f;           CHECK(!tars_gait_in_range(&g), "f 3.5 Hz rejected");
        g = tars_gait_fwd_v7; g.a1 = 0.9f;          CHECK(!tars_gait_in_range(&g), "hip a1 0.9 rad rejected");
        g = tars_gait_fwd_v7; g.c1 = -0.05f;        CHECK(!tars_gait_in_range(&g), "lift c1 50 mm rejected");
        g = tars_gait_fwd_v7; g.kp_h = 20.0f;       CHECK(!tars_gait_in_range(&g), "kp_h 20 rejected");
        g = tars_gait_fwd_v7; g.kd_z = NAN;         CHECK(!tars_gait_in_range(&g), "NaN rejected");
    }

    /* standing assist hook (learned push recovery): used only while standing upright, never while walking / fallen, plain stand when off */
    {
        const float upr[3] = {0.0f, 0.0f, 1.0f};
        const float tilted[3] = {0.0f, 0.5f, 0.7f};          /* up[2] = 0.7: below stand_up_min (0.8), above the fall threshold (0.5) */
        const float zero3[3] = {0, 0, 0};
        float q[2], z;
        setup();
        C.stand_reset = as_reset; C.stand_step = as_step;
        g_as_reset = g_as_step = 0;
        for (int i = 0; i < 50; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(g_as_step == 0 && fabsf(q[0]) < 1e-6f, "assist off (stand_on = 0): plain stand, callback not called (%d)", g_as_step);
        C.stand_on = 1;
        for (int i = 0; i < 50; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(g_as_step == 50 && g_as_reset == 1, "assist on: stepped 50 times, reset once (%d, %d)", g_as_step, g_as_reset);
        CHECK(fabsf(q[0] - 0.1f) < 1e-6f && fabsf(q[1] + 0.2f) < 1e-6f && fabsf(z - 0.003f) < 1e-6f, "assist output used: %f %f %f", q[0], q[1], z);
        for (int i = 0; i < 10; i++) tars_ctrl_step(&C, tilted, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(g_as_step == 50, "tilted (up[2] 0.7): assist not called");
        CHECK(fabsf(q[0]) < 0.1f, "plain stand slews back towards the nominal pose: %f", q[0]);
        for (int i = 0; i < 10; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(g_as_reset == 2, "assist restarts (reset again) after it was interrupted (%d)", g_as_reset);
        const int before = g_as_step;
        for (int i = 0; i < 100; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 20.0f, 0.0f, q, &z);       /* walking command */
        CHECK(C.state == TARS_S_WALK_F && g_as_step == before, "walking: assist not called (state %d)", C.state);
        C.stand_step = NULL;
        setup();
        C.stand_on = 1;                                       /* on but no callback: plain stand */
        for (int i = 0; i < 20; i++) tars_ctrl_step(&C, upr, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(fabsf(q[0]) < 1e-6f, "assist on without a callback: plain stand");
        setup();
        C.stand_reset = as_reset; C.stand_step = as_step; C.stand_on = 1;
        const float on_back2[3] = {1.0f, 0.0f, 0.0f};
        for (int i = 0; i < 30; i++) tars_ctrl_step(&C, on_back2, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_FALLEN || C.state == TARS_S_GETUP1, "lying on the back: fall detected with the assist on (state %d)", C.state);
    }

    /* ---------------------------------------------------------------------------------------- show motions (MOTION state) */
    {
        const float upr[3] = {0.0f, 0.0f, 1.0f};
        const float zero3[3] = {0, 0, 0};
        float q[2] = {0, 0}, z = 0;
        const tars_motion_t *bow = &tars_motions[1];
        const tars_motion_t *nod = &tars_motions[0];
        CHECK(TARS_MOTION_COUNT >= 3 && !strcmp(nod->name, "nod") && !strcmp(bow->name, "bow") && !strcmp(tars_motions[2].name, "headbang"),
              "the first three motions keep their names / order (the real-robot tests used them): %d motions", TARS_MOTION_COUNT);
        {   /* names: unique, short enough for the shell lines; faces: only the emotions have one, and it is one of the screen's emoji (the .eaf assets in main/boards/tars/emoji) */
            static const char *const emoji[] = {"angry", "confident", "confused", "cool", "crying", "delicious", "embarrassed", "happy", "kiss", "laughing", "loving", "neutral", "relaxed",
                                                "sad", "shocked", "silly", "sleepy", "surprised", "thinking", "winking"};
            int names_len = 0, faces = 0;
            for (int k = 0; k < TARS_MOTION_COUNT; k++) {
                const tars_motion_t *m = &tars_motions[k];
                names_len += (int)strlen(m->name) + 1;
                for (int j = 0; j < k; j++) CHECK(strcmp(m->name, tars_motions[j].name), "duplicate motion name %s", m->name);
                CHECK(strlen(m->name) >= 2 && strlen(m->name) <= 12, "%s: name length", m->name);
                if (k < 3 || k >= 15) CHECK(m->face == NULL, "%s: a plain show motion / an arm motion has no face", m->name);
                CHECK(strcmp(m->name, "sneeze"), "the sneeze is not in the firmware (it tipped the real robot over at full speed)");
                if (m->face) {
                    int ok = 0;
                    for (size_t e = 0; e < sizeof emoji / sizeof emoji[0]; e++) ok |= !strcmp(m->face, emoji[e]);
                    CHECK(ok, "%s: face '%s' is not a screen emoji", m->name, m->face);
                    faces++;
                }
            }
            CHECK(names_len < 200, "the name list must fit the shell's buffers: %d chars", names_len);
            CHECK(faces == 12, "every emotion has a face name (%d of 12)", faces);
            CHECK(TARS_MOTION_COUNT == 21 && !strcmp(tars_motions[3].name, "happy") && !strcmp(tars_motions[14].name, "impatient") && !strcmp(tars_motions[15].name, "wave") && !strcmp(tars_motions[20].name, "twist"), "the table order: 3 show motions, 12 emotions, 6 arm motions (%d)", TARS_MOTION_COUNT);
        }

        /* the generated tables: start and end at the nominal pose, never more than 8 rad/s / 0.25 m/s / the servo ranges, sane tilt limits */
        for (int k = 0; k < TARS_MOTION_COUNT; k++) {
            const tars_motion_t *m = &tars_motions[k];
            float vh = 0.0f, vz = 0.0f, qmax = 0.0f, zmax = 0.0f, peak = 0.0f;
            for (int i = 0; i < m->n; i++) {
                if (fabsf(m->hl[i] * 1e-3f) > qmax) qmax = fabsf(m->hl[i] * 1e-3f);
                if (fabsf(m->hr[i] * 1e-3f) > qmax) qmax = fabsf(m->hr[i] * 1e-3f);
                if (fabsf(m->z[i] * 1e-4f) > zmax) zmax = fabsf(m->z[i] * 1e-4f);
                if (i) {
                    float d = fabsf((m->hl[i] - m->hl[i - 1]) * 1e-3f) / m->dt; if (d > vh) vh = d;
                    d = fabsf((m->hr[i] - m->hr[i - 1]) * 1e-3f) / m->dt; if (d > vh) vh = d;
                    d = fabsf((m->z[i] - m->z[i - 1]) * 1e-4f) / m->dt; if (d > vz) vz = d;
                }
            }
            peak = qmax * 57.29578f;
            float zmin = 0.0f;
            for (int i = 0; i < m->n; i++) if (m->z[i] * 1e-4f < zmin) zmin = m->z[i] * 1e-4f;
            CHECK(m->n >= 200 && m->dt > 0.019f && m->dt < 0.021f, "%s: %d samples every %.3f s", m->name, m->n, m->dt);
            CHECK(m->hl[0] == 0 && m->hr[0] == 0 && m->z[0] == 0 && m->hl[m->n - 1] == 0 && m->hr[m->n - 1] == 0 && m->z[m->n - 1] == 0, "%s starts and ends at the nominal pose", m->name);
            CHECK(vh <= 8.05f && vz <= 0.26f, "%s: hip speed %.2f rad/s, lift speed %.3f m/s", m->name, vh, vz);
            CHECK(qmax <= 2.4f && zmax <= 0.0141f, "%s: hip %.2f rad, lift %.4f m inside the servo ranges", m->name, qmax, zmax);
            if (zmin < -0.006f) {                 /* the arms stand on the floor (lift extended): the torso pitches with the hips, its limit must clear the peak pitch */
                CHECK(m->tilt_limit_deg >= peak + 15.0f && m->tilt_limit_deg <= 80.0f, "%s: tilt limit %.0f deg vs peak pitch %.0f deg", m->name, m->tilt_limit_deg, peak);
            } else {                              /* arms retracted, the robot stands on its torso base: the torso must stay upright, the arms swing (hips up to 2.4 rad) */
                CHECK(m->tilt_limit_deg >= 15.0f && m->tilt_limit_deg <= 30.0f, "%s: an arm motion keeps the torso upright, limit %.0f deg", m->name, m->tilt_limit_deg);
                CHECK(zmin > -0.0011f, "%s: an arm motion never extends the arms (lift %.4f)", m->name, zmin);
            }
        }

        /* requests: bad arguments, not while lying, one at a time */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        CHECK(tars_ctrl_motion_start(&C, NULL, 1.0f) == -1, "NULL motion refused");
        { tars_motion_t bad = *bow; bad.n = 1; CHECK(tars_ctrl_motion_start(&C, &bad, 1.0f) == -1, "one-sample table refused"); }
        CHECK(tars_ctrl_motion_start(&C, bow, 1.0f) == 0, "accepted at STAND");
        CHECK(tars_ctrl_motion_start(&C, nod, 1.0f) == -3, "a second request is refused (one at a time)");
        fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_MOTION && C.motion == bow && C.motion_req == NULL, "starts at the next step (state %d)", C.state);
        CHECK(tars_ctrl_motion_start(&C, nod, 1.0f) == -3, "refused while a motion is running");
        CHECK(fabsf(q[0]) < 1e-6f && fabsf(z) < 1e-6f, "the first step keeps the pose of the stand");
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        run((const float[3]){1.0f, 0.0f, 0.0f}, 0.0f, 1.0);                       /* lying on the back: FALLEN / GETUP */
        CHECK(C.state >= TARS_S_FALLEN, "lying: state %d", C.state);
        CHECK(tars_ctrl_motion_start(&C, bow, 1.0f) == -3, "refused while lying / getting up");

        /* the output is the table: after j steps of 20 ms the targets are sample j (50 Hz = the table rate) */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        fstep(0.02f, 0.0f, 0.0f, q, &z);                 /* start step */
        int j_ok = 1;
        for (int j = 1; j <= 300; j++) {
            fstep(0.02f, 0.0f, 0.0f, q, &z);
            if (fabsf(q[0] - bow->hl[j] * 1e-3f) > 1e-4f || fabsf(q[1] - bow->hr[j] * 1e-3f) > 1e-4f || fabsf(z - bow->z[j] * 1e-4f) > 1e-6f) { j_ok = 0; printf("  sample %d: %f %f %f vs %d %d\n", j, q[0], q[1], z, bow->hl[j], bow->hr[j]); break; }
        }
        CHECK(j_ok, "the targets are the table samples (bow, steps 1..300)");
        CHECK(C.state == TARS_S_MOTION && fabsf(q[0]) < 1e-6f + fabsf(bow->hl[300] * 1e-3f), "still playing at 6 s");
        for (int j = 301; j < 420; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && C.motion_end == 1 && C.motion == NULL, "played to the end: STAND (state %d end %d)", C.state, C.motion_end);
        CHECK(fabsf(q[0]) < 1e-6f && fabsf(q[1]) < 1e-6f && fabsf(z) < 1e-6f && fabsf(C.hc) < 1e-6f && fabsf(C.zc) < 1e-6f, "back at the nominal pose: %f %f %f", q[0], q[1], z);
        for (int j = 0; j < 100; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && fabsf(q[0]) < 1e-6f, "stays in STAND afterwards");

        /* speed: half speed takes twice as long; a speed outside 0.3..1 is clamped */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 0.5f);
        for (int j = 0; j < 51 + 280; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_MOTION && C.motion_t > 3.2f && C.motion_t < 3.4f, "half speed: after 6.6 s the table is at %.2f s (state %d)", C.motion_t, C.state);
        for (int j = 0; j < 300; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND, "half speed: done after about twice the time (state %d)", C.state);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 5.0f);
        CHECK(C.motion_req_speed == 1.0f, "speed above 1 clamped to 1 (%f)", C.motion_req_speed);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 0.0f);
        CHECK(C.motion_req_speed == 0.3f, "speed below 0.3 clamped to 0.3 (%f)", C.motion_req_speed);

        /* at the 500 Hz controller rate the table is interpolated linearly */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        fstep(0.002f, 0.0f, 0.0f, q, &z);
        for (int j = 0; j < 5 * 125 + 5; j++) fstep(0.002f, 0.0f, 0.0f, q, &z);       /* 5 * 125 + 5 steps of 2 ms: 1.26 s = table index 63 */
        {
            const float tt = (5 * 125 + 5) * 0.002f, pos = tt / bow->dt;
            const int i = (int)pos;
            const float f = pos - (float)i;
            const float want = (bow->hl[i] * (1.0f - f) + bow->hl[i + 1] * f) * 1e-3f;
            CHECK(fabsf(q[0] - want) < 2e-4f, "500 Hz: %f vs interpolated %f", q[0], want);
        }

        /* a walking command cuts it short: the hips go home FIRST, the lift last (the torso must be upright before the arms retract) */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 175; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);       /* 3.5 s: held bow, lift extended */
        CHECK(C.state == TARS_S_MOTION && q[0] > 0.9f && z < -0.013f, "bow held: hip %.2f lift %.4f", q[0], z);
        float z_when_hips_home = 9.0f;
        int steps_home = 0;
        for (int j = 0; j < 200 && C.state == TARS_S_MOTION; j++) {
            fstep(0.02f, 20.0f, 0.0f, q, &z);
            steps_home++;
            if (fabsf(q[0]) < 0.011f && z_when_hips_home > 1.0f) z_when_hips_home = z;
        }
        CHECK(C.motion_end == 3, "ended by the walking command (end %d)", C.motion_end);
        CHECK(z_when_hips_home < -0.013f, "the lift was still extended (%.4f) when the hips were home", z_when_hips_home);
        CHECK(steps_home * 0.02f > 0.5f && steps_home * 0.02f < 1.0f, "home in %.2f s (hips 2 rad/s, then the lift 0.05 m/s)", steps_home * 0.02f);
        for (int j = 0; j < 100; j++) fstep(0.02f, 20.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_WALK_F, "then it walks (state %d)", C.state);

        /* stop() and withdrawing a request */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 1.0f);
        for (int j = 0; j < 100; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        tars_ctrl_motion_stop(&C);
        for (int j = 0; j < 100; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && C.motion_end == 2 && fabsf(q[0]) < 1e-6f && fabsf(z) < 1e-6f, "stopped: STAND at the nominal pose (state %d end %d)", C.state, C.motion_end);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 1.0f);
        tars_ctrl_motion_stop(&C);
        fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && C.motion_req == NULL && C.motion_end == 2, "a withdrawn request never starts (state %d)", C.state);

        /* the request waits for a still, upright, non-walking robot */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        for (int j = 0; j < 20; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        const float moving[3] = {0.0f, 1.5f, 0.0f};                                // 1.5 rad/s: being handled
        tars_ctrl_motion_start(&C, nod, 1.0f);
        for (int j = 0; j < 50; j++) tars_ctrl_step(&C, upr, moving, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && C.motion_req == nod, "not started while the robot moves (1.5 rad/s)");
        const float lean[3] = {0.0f, 0.0f, 0.93f};                                 // 21 deg: not upright enough
        for (int j = 0; j < 50; j++) tars_ctrl_step(&C, lean, zero3, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND, "not started while leaning 21 deg");
        fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND || C.state == TARS_S_MOTION, "state %d", C.state);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 1.0f);
        for (int j = 0; j < 160; j++) tars_ctrl_step(&C, upr, moving, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_STAND && C.motion_req == NULL && C.motion_end == 5, "a request that gets no still stand within 3 s is dropped (end %d)", C.motion_end);

        /* a request made while walking waits for the walk to end; a walk that never ends drops it after 3 s */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        for (int j = 0; j < 100; j++) fstep(0.02f, 20.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_WALK_F, "walking");
        CHECK(tars_ctrl_motion_start(&C, nod, 1.0f) == 0, "accepted while walking");
        for (int j = 0; j < 100; j++) fstep(0.02f, 20.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_WALK_F && C.motion_req == nod, "the walk goes on: still waiting after 2 s");
        for (int j = 0; j < 55; j++) fstep(0.02f, 20.0f, 0.0f, q, &z);
        CHECK(C.motion_req == NULL && C.motion_end == 5 && C.state == TARS_S_WALK_F, "dropped after 3 s of walking (end %d)", C.motion_end);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        for (int j = 0; j < 100; j++) fstep(0.02f, 20.0f, 0.0f, q, &z);
        tars_ctrl_motion_start(&C, nod, 1.0f);
        int seen_stand = 0, seen_motion = 0;
        for (int j = 0; j < 300; j++) {
            fstep(0.02f, 0.0f, 0.0f, q, &z);          /* the walking command stops */
            if (C.state == TARS_S_STAND) seen_stand = 1;
            if (C.state == TARS_S_MOTION) { seen_motion = 1; break; }
        }
        CHECK(seen_stand && seen_motion, "walk ended -> STAND -> MOTION (stand %d motion %d)", seen_stand, seen_motion);

        /* the absolute tilt limit: the usual 60 deg fall detection is off during a motion, the motion's own limit (the bow's: 75 deg) ends it as a fall.
         * The held bow (table time 2.2..3.4 s) has the hips at 54 deg; the attitude fed is "hips + extra", so that the leg guard (30 deg) stays quiet below. */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);                 /* table time 2.4 s */
        CHECK(C.state == TARS_S_MOTION && q[0] > 0.94f, "held bow: hip %.2f", q[0]);
        for (int j = 0; j < 15; j++) fstep_extra(0.15f, 0.02f, 0.0f, 0.0f, q, &z);               /* 54 + 8.6 = 63 deg: beyond the walking limit (60), inside the bow's 75 */
        CHECK(C.state == TARS_S_MOTION, "63 deg during the bow is fine (state %d)", C.state);
        for (int j = 0; j < 8; j++) fstep_extra(0.44f, 0.02f, 0.0f, 0.0f, q, &z);                /* 54 + 25 = 79 deg: beyond the bow's limit for 0.16 s */
        CHECK(C.state == TARS_S_MOTION, "79 deg for 0.16 s: not yet, the limit needs 0.2 s (state %d)", C.state);
        for (int j = 0; j < 3; j++) fstep_extra(0.15f, 0.02f, 0.0f, 0.0f, q, &z);
        for (int j = 0; j < 8; j++) fstep_extra(0.44f, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_MOTION, "the 0.2 s have to be continuous (state %d)", C.state);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        const float hip_then = q[0];
        int fell_at = -1;
        for (int j = 0; j < 40; j++) {
            fstep_extra(0.44f, 0.02f, 0.0f, 0.0f, q, &z);                                         /* 79 deg for good */
            if (C.state == TARS_S_FALLEN) { fell_at = j; break; }
        }
        CHECK(fell_at >= 9 && fell_at <= 11 && C.motion_end == 4 && C.motion == NULL, "79 deg for 0.2 s: FALLEN after %d steps (end %d = tilt limit)", fell_at, C.motion_end);
        CHECK(fabsf(q[0] - hip_then) < 0.4f, "the targets continue from where the motion was (hip %.2f -> %.2f)", hip_then, q[0]);

        /* the leg guard: while the lift is extended the torso pitch follows the hips; 30 deg apart for 0.15 s = the whole robot rocks over its toes (the full-speed sneeze, 63 deg) */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        for (int j = 0; j < 60; j++) fstep_extra(0.35f, 0.02f, 0.0f, 0.0f, q, &z);               /* 20 deg apart (below the 30 deg guard, inside the tilt limit: 54 + 20 = 74 < 75) */
        CHECK(C.state == TARS_S_MOTION, "legs 20 deg off: no guard (state %d)", C.state);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        fell_at = -1;
        for (int j = 0; j < 40; j++) {
            fstep_extra(0.61f, 0.02f, 0.0f, 0.0f, q, &z);                                         /* 35 deg apart */
            if (C.state == TARS_S_FALLEN) { fell_at = j; break; }
        }
        CHECK(fell_at >= 6 && fell_at <= 8 && C.motion_end == 6 && C.motion == NULL, "legs 35 deg off for 0.15 s: FALLEN after %d steps (end %d = legs tilted)", fell_at, C.motion_end);
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;                                              /* a short blip does not count */
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        for (int k = 0; k < 6; k++) {
            for (int j = 0; j < 5; j++) fstep_extra(0.61f, 0.02f, 0.0f, 0.0f, q, &z);            /* 0.1 s off ... */
            fstep(0.02f, 0.0f, 0.0f, q, &z);                                                /* ... one good step */
        }
        CHECK(C.state == TARS_S_MOTION, "blips of 0.1 s do not add up (state %d)", C.state);
        {   /* the guard needs the legs on the floor: a motion with the arms retracted (lift +14 mm, standing on the torso: the arms swing freely) is not watched this way */
            static int16_t flat_h[100], flat_z[100];
            for (int k = 0; k < 100; k++) { flat_h[k] = 0; flat_z[k] = 140; }
            const tars_motion_t ready = {"ready_test", 100, 0.02f, 75.0f, flat_h, flat_h, flat_z, NULL};
            setup(); q[0] = q[1] = 0.0f; z = 0.0f;
            tars_ctrl_motion_start(&C, &ready, 1.0f);
            for (int j = 0; j < 1 + 20; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
            for (int j = 0; j < 40; j++) fstep_extra(0.61f, 0.02f, 0.0f, 0.0f, q, &z);        /* 35 deg off for 0.8 s */
            CHECK(C.state == TARS_S_MOTION && z > 0.013f, "no leg guard while the arms are retracted (state %d, lift %.4f)", C.state, z);
        }
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;                                              /* nor while the hips go home after a stop */
        tars_ctrl_motion_start(&C, bow, 1.0f);
        for (int j = 0; j < 1 + 120; j++) fstep(0.02f, 0.0f, 0.0f, q, &z);
        tars_ctrl_motion_stop(&C);
        for (int j = 0; j < 12; j++) fstep_extra(0.61f, 0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_MOTION && C.motion_leg_t == 0.0f, "homing after a stop: the guard is off (state %d)", C.state);
        /* a request does not survive a fall */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, nod, 1.0f);
        for (int j = 0; j < 60; j++) tars_ctrl_step(&C, (const float[3]){1.0f, 0.0f, 0.0f}, moving, 0.02f, 0.0f, 0.0f, q, &z);   /* tipped over while it waited (still handled) */
        CHECK(C.state >= TARS_S_FALLEN && C.motion_req == NULL, "request dropped by the fall (state %d)", C.state);
        /* reset clears everything */
        setup(); q[0] = q[1] = 0.0f; z = 0.0f;
        tars_ctrl_motion_start(&C, bow, 1.0f);
        fstep(0.02f, 0.0f, 0.0f, q, &z);
        CHECK(C.state == TARS_S_MOTION, "running");
        tars_ctrl_reset(&C);
        CHECK(C.state == TARS_S_STAND && C.motion == NULL && C.motion_req == NULL, "reset: STAND, nothing pending");
        CHECK(!strcmp(tars_state_name(TARS_S_MOTION), "MOTION") && !strcmp(tars_state_name(TARS_S_GETUP_NN), "GETUP_NN"), "state names");
    }

    printf("test_ctrl: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
