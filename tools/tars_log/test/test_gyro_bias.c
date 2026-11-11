/* Unit tests of the gyro zero-rate offset estimator in tars_rl_core.c (firmware-only part of the learned runtime). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tars_rl_core.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                                  \
    do {                                                                             \
        g_checks++;                                                                  \
        if (!(cond)) {                                                               \
            g_fail++;                                                                \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
        }                                                                            \
    } while (0)

static unsigned g_rng = 2463534242u;
static float rnd(void)                         /* uniform in [-1, 1) */
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static const float kBias[3] = {0.045f, 0.166f, -0.012f};     /* the second Tars: 2.6 / 9.5 / -0.7 deg/s */

static void still_sample(float acc[3], float gyro[3], float noise, const float bias[3])
{
    acc[0] = 0.35f + 0.02f * rnd();
    acc[1] = 9.89f + 0.02f * rnd();
    acc[2] = 0.69f + 0.02f * rnd();
    for (int k = 0; k < 3; k++) gyro[k] = bias[k] + noise * rnd();
}

static void test_start_on_a_still_robot(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3], gyro[3], m[3] = {0, 0, 0};
    int started_at = -1;
    for (int i = 1; i <= 100 && started_at < 0; i++) {
        still_sample(acc, gyro, 0.01f, kBias);
        if (tars_gyro_bias_start(&b, acc, gyro, m)) started_at = i;
    }
    CHECK(started_at == 30);                                   /* exactly after the 30 sample window */
    for (int k = 0; k < 3; k++) CHECK(fabsf(b.bias[k] - kBias[k]) < 0.01f);
    CHECK(fabsf(m[0] - 0.35f) < 0.05f && fabsf(m[1] - 9.89f) < 0.05f && fabsf(m[2] - 0.69f) < 0.05f);
}

static void test_no_start_while_handled(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3], gyro[3], m[3];
    int started = 0;
    for (int i = 0; i < 400; i++) {                            /* hand-held: the gyro wanders by +-0.4 rad/s (std 0.23) */
        still_sample(acc, gyro, 0.4f, kBias);
        started += tars_gyro_bias_start(&b, acc, gyro, m);
    }
    CHECK(started == 0);
    int at = -1;
    for (int i = 1; i <= 100 && at < 0; i++) {                 /* put down: starts one window later */
        still_sample(acc, gyro, 0.01f, kBias);
        if (tars_gyro_bias_start(&b, acc, gyro, m)) at = i;
    }
    CHECK(at == 30);
    for (int k = 0; k < 3; k++) CHECK(fabsf(b.bias[k] - kBias[k]) < 0.01f);
}

static void test_no_start_when_not_gravity_or_fast_spin(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3] = {0, 0, 14.0f}, gyro[3] = {0.1f, 0.1f, 0.1f}, m[3];
    int started = 0;
    for (int i = 0; i < 200; i++) started += tars_gyro_bias_start(&b, acc, gyro, m);    /* |a| = 14: being shaken */
    acc[2] = 9.81f;
    gyro[0] = 0.6f;                                                                       /* > 0.5 rad/s: spinning */
    for (int i = 0; i < 200; i++) started += tars_gyro_bias_start(&b, acc, gyro, m);
    CHECK(started == 0);
}

static void test_apply_removes_the_offset_and_tracks_drift(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3], gyro[3], m[3];
    for (int i = 0; i < 40; i++) {
        still_sample(acc, gyro, 0.01f, kBias);
        if (tars_gyro_bias_start(&b, acc, gyro, m)) break;
    }
    /* 2 minutes standing still while the offset drifts by +0.1 deg/s per second (the IMU warms up): the output must stay near zero */
    float drift = 0.0f, worst = 0.0f, sum_tail = 0.0f;
    int tail_n = 0;
    for (int i = 0; i < 12000; i++) {
        drift += 0.1f * 0.0174533f * 0.01f;
        float bias_now[3] = {kBias[0] + drift, kBias[1] + drift, kBias[2] + drift};
        still_sample(acc, gyro, 0.01f, bias_now);
        tars_gyro_bias_apply(&b, acc, gyro);
        if (i > 3000 && fabsf(gyro[1]) > worst) worst = fabsf(gyro[1]);
        if (i >= 11000) { sum_tail += gyro[1]; tail_n++; }
    }
    CHECK(worst < 0.02f);                                       /* noise (+-0.01) plus the tracking lag (<0.004): never above 1.1 deg/s after the first 30 s */
    CHECK(fabsf(sum_tail / (float)tail_n) < 0.004f);
}

static void test_walking_does_not_pollute_the_estimate(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3], gyro[3], m[3];
    for (int i = 0; i < 40; i++) {
        still_sample(acc, gyro, 0.01f, kBias);
        if (tars_gyro_bias_start(&b, acc, gyro, m)) break;
    }
    const float before1 = b.bias[1];
    for (int i = 0; i < 3000; i++) {                            /* 30 s of walking: body rocks +-0.5 rad/s at 1.25 Hz, |a| swings by +-2 m/s^2 */
        const float ph = 6.2831853f * 1.25f * (float)i * 0.01f;
        acc[0] = 0.35f + 2.0f * sinf(ph); acc[1] = 9.89f + 1.0f * cosf(ph); acc[2] = 0.69f;
        for (int k = 0; k < 3; k++) gyro[k] = kBias[k] + 0.5f * sinf(ph + (float)k);
        tars_gyro_bias_apply(&b, acc, gyro);
    }
    CHECK(fabsf(b.bias[1] - before1) < 0.003f);
    CHECK(b.static_n == 0 || b.static_n < 100);
}

static void test_adaptation_waits_one_second_after_motion(void)
{
    tars_gyro_bias_t b;
    tars_gyro_bias_reset(&b);
    float acc[3], gyro[3], m[3];
    for (int i = 0; i < 40; i++) {
        still_sample(acc, gyro, 0.01f, kBias);
        if (tars_gyro_bias_start(&b, acc, gyro, m)) break;
    }
    const float b0 = b.bias[1];
    for (int i = 0; i < 100; i++) {                             /* exactly 1 s of static samples with a +1 deg/s step: bias must not move yet */
        float bias_now[3] = {kBias[0], kBias[1] + 0.0174533f, kBias[2]};
        still_sample(acc, gyro, 0.001f, bias_now);
        tars_gyro_bias_apply(&b, acc, gyro);
    }
    CHECK(fabsf(b.bias[1] - b0) < 1e-6f);
    for (int i = 0; i < 400; i++) {
        float bias_now[3] = {kBias[0], kBias[1] + 0.0174533f, kBias[2]};
        still_sample(acc, gyro, 0.001f, bias_now);
        tars_gyro_bias_apply(&b, acc, gyro);
    }
    CHECK(b.bias[1] > b0 + 0.004f);                             /* now it follows */
}

static void test_tars_upright_axes(void)
{
    const float up[3] = {0.33f, 9.89f, 0.69f}; /* measured on the second TARS */
    const float forward[3] = {0.0f, 0.0f, -9.8f};
    float rotation[9];
    CHECK(tars_imu_axes_from_refs(up, forward, rotation));
    CHECK(rotation[2] < -0.99f && rotation[3] < -0.99f && rotation[7] > 0.99f);

    /* The same mounting with a 30-degree forward lean must read as nose-down. */
    const float nose_down[3] = {0.0f, 8.49f, 4.9f};
    const float body_x = rotation[0] * nose_down[0] + rotation[1] * nose_down[1] + rotation[2] * nose_down[2];
    CHECK(body_x < -4.0f);
    CHECK(!tars_imu_axes_from_refs(up, up, rotation));
}

int main(void)
{
    test_start_on_a_still_robot();
    test_no_start_while_handled();
    test_no_start_when_not_gravity_or_fast_spin();
    test_apply_removes_the_offset_and_tracks_drift();
    test_walking_does_not_pollute_the_estimate();
    test_adaptation_waits_one_second_after_motion();
    test_tars_upright_axes();
    printf("test_gyro_bias: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
