/* Unit tests of tars_log_core.c (portable part of the Tars log tool). Build + run: make test_core && ./test_core */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tars_log_core.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                                  \
    do {                                                                             \
        g_checks++;                                                                  \
        if (!(cond)) {                                                               \
            g_fail++;                                                                \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
        }                                                                            \
    } while (0)

static void test_numbers(void)
{
    char b[32];
    size_t n;
    n = tars_log_put_i32(b, 0); b[n] = 0; CHECK(!strcmp(b, "0"));
    n = tars_log_put_i32(b, -1234567); b[n] = 0; CHECK(!strcmp(b, "-1234567"));
    n = tars_log_put_i32(b, 2147483647); b[n] = 0; CHECK(!strcmp(b, "2147483647"));
    n = tars_log_put_i32(b, (-2147483647 - 1)); b[n] = 0; CHECK(!strcmp(b, "-2147483648"));
    n = tars_log_put_milli(b, 12345, 3); b[n] = 0; CHECK(!strcmp(b, "12.345"));
    n = tars_log_put_milli(b, -5, 3); b[n] = 0; CHECK(!strcmp(b, "-0.005"));
    n = tars_log_put_milli(b, -4, 2); b[n] = 0; CHECK(!strcmp(b, "0.00"));        /* rounds to zero: no "-0.00" */
    n = tars_log_put_milli(b, -5, 2); b[n] = 0; CHECK(!strcmp(b, "-0.01"));
    n = tars_log_put_milli(b, 9996, 2); b[n] = 0; CHECK(!strcmp(b, "10.00"));
    n = tars_log_put_milli(b, 7, 0); b[n] = 0; CHECK(!strcmp(b, "0"));
    n = tars_log_put_milli(b, 500, 0); b[n] = 0; CHECK(!strcmp(b, "1"));
    n = tars_log_put_milli(b, 1500, 1); b[n] = 0; CHECK(!strcmp(b, "1.5"));

    int32_t v = 0;
    CHECK(tars_log_parse_i32("42", &v) && v == 42);
    CHECK(tars_log_parse_i32("-17", &v) && v == -17);
    CHECK(tars_log_parse_i32("+8", &v) && v == 8);
    CHECK(!tars_log_parse_i32("", &v));
    CHECK(!tars_log_parse_i32("-", &v));
    CHECK(!tars_log_parse_i32("12x", &v));
    CHECK(!tars_log_parse_i32("1.5", &v));
    CHECK(!tars_log_parse_i32("99999999999", &v));
    CHECK(tars_log_parse_i32("-2147483648", &v) && v == (-2147483647 - 1));
    CHECK(tars_log_parse_milli("1.5", &v) && v == 1500);
    CHECK(tars_log_parse_milli("-0.25", &v) && v == -250);
    CHECK(tars_log_parse_milli("3", &v) && v == 3000);
    CHECK(tars_log_parse_milli(".5", &v) && v == 500);
    CHECK(tars_log_parse_milli("0.2", &v) && v == 200);
    CHECK(tars_log_parse_milli("1.23456", &v) && v == 1234);
    CHECK(!tars_log_parse_milli("1.2.3", &v));
    CHECK(!tars_log_parse_milli(".", &v));
    CHECK(!tars_log_parse_milli("abc", &v));

    char line[] = "  step   1\t40  600 ";
    char *argv[8];
    int argc = tars_log_tokenize(line, argv, 8);
    CHECK(argc == 4 && !strcmp(argv[0], "step") && !strcmp(argv[1], "1") && !strcmp(argv[2], "40") && !strcmp(argv[3], "600"));
    char l2[] = "a b c d e f";
    CHECK(tars_log_tokenize(l2, argv, 3) == 3);
    char l3[] = "   ";
    CHECK(tars_log_tokenize(l3, argv, 3) == 0);
}

static void test_lines(void)
{
    tars_log_ev_t e;
    memset(&e, 0, sizeof e);
    e.type = TLE_IMU;
    e.t_us = 123456;
    for (int i = 0; i < 12; i++) e.v[i] = (int16_t)(i * 1000 - 5000);
    char out[200];
    size_t n = tars_log_format_event(&e, out, sizeof out);
    CHECK(n > 0 && out[n - 1] == '\n');
    out[n - 1] = 0;
    CHECK(tars_log_check_line(out));
    CHECK(!strncmp(out, "@I,123456,-5000,-4000,-3000,-2000,-1000,0,1000,2000,3000,4000,5000,6000*", 70));
    out[10] ^= 1;                                      /* corrupt one character */
    CHECK(!tars_log_check_line(out));
    out[10] ^= 1;
    CHECK(tars_log_check_line(out));

    e.type = TLE_SERVO; e.id = 2; e.v[0] = 512; e.v[1] = -3; e.v[2] = 99; e.v[3] = 700; e.t_us = -5;
    n = tars_log_format_event(&e, out, sizeof out);
    out[n - 1] = 0;
    CHECK(!strncmp(out, "@S,-5,2,512,-3,99,700*", 22) && tars_log_check_line(out));

    e.type = TLE_CMD; e.id = 1; e.v[0] = 430; e.v[1] = 540; e.v[2] = 492; e.v[3] = 0; e.t_us = 0;
    n = tars_log_format_event(&e, out, sizeof out);
    out[n - 1] = 0;
    CHECK(!strncmp(out, "@C,0,1,430,540,492,0*", 21) && tars_log_check_line(out));

    e.type = TLE_VOLT; e.v[0] = 742; e.t_us = 1000000;
    n = tars_log_format_event(&e, out, sizeof out);
    out[n - 1] = 0;
    CHECK(!strncmp(out, "@V,1000000,742*", 15) && tars_log_check_line(out));

    memset(e.v, 0, sizeof e.v);
    e.type = TLE_MSG; e.t_us = 77;
    memcpy(e.v, "state STAND*->WALK, ok", 22);
    n = tars_log_format_event(&e, out, sizeof out);
    out[n - 1] = 0;
    CHECK(!strncmp(out, "@M,77,state STAND ->WALK, ok*", 29) && tars_log_check_line(out));   /* '*' replaced */
    e.type = 'Z';
    CHECK(tars_log_format_event(&e, out, sizeof out) == 0);
    CHECK(tars_log_format_event(&e, out, 10) == 0);
    CHECK(!tars_log_check_line("@I,1,2*00"));
    CHECK(!tars_log_check_line("I,1,2*00"));
    CHECK(!tars_log_check_line("@I,1,2"));
}

static void test_ring_basic(void)
{
    tars_log_ev_t *mem = calloc(16, sizeof *mem);
    tars_log_ring_t r;
    tars_log_ring_init(&r, mem, 16);
    tars_log_ring_reset(&r, false, 1000);
    int16_t v[3] = {1, 2, 3};
    for (int i = 0; i < 20; i++) {
        v[0] = (int16_t)i;
        bool ok = tars_log_ring_push(&r, 1000 + i * 10, TLE_VOLT, 0, v, 3);
        CHECK(ok == (i < 16));
    }
    CHECK(tars_log_ring_head(&r) == 16);
    CHECK(r.dropped == 4);
    CHECK(tars_log_ring_first(&r) == 0);
    tars_log_ev_t e;
    for (uint32_t i = 0; i < 16; i++) {
        CHECK(tars_log_ring_get(&r, i, &e) == 1);
        CHECK(e.v[0] == (int16_t)i && e.t_us == (int32_t)(i * 10) && e.v[3] == 0);
    }
    /* reset clears */
    tars_log_ring_reset(&r, false, 5000);
    CHECK(tars_log_ring_head(&r) == 0);
    CHECK(tars_log_ring_get(&r, 0, &e) == 0);
    /* overwrite mode */
    tars_log_ring_reset(&r, true, 0);
    for (int i = 0; i < 40; i++) {
        v[0] = (int16_t)i;
        CHECK(tars_log_ring_push(&r, i, TLE_VOLT, 0, v, 3));
    }
    CHECK(tars_log_ring_head(&r) == 40);
    CHECK(tars_log_ring_first(&r) == 24);
    CHECK(tars_log_ring_get(&r, 10, &e) == -1);
    CHECK(tars_log_ring_get(&r, 24, &e) == 1 && e.v[0] == 24);
    CHECK(tars_log_ring_get(&r, 39, &e) == 1 && e.v[0] == 39);
    CHECK(tars_log_ring_get(&r, 40, &e) == 0);
    free(mem);
}

/* concurrency: 4 producers write self-checking events into a small overwrite ring while a consumer reads */
#define NPROD 4
#define PER_PROD 60000
static tars_log_ring_t g_ring;
static volatile int g_done = 0;
static void *producer(void *arg)
{
    const int id = (int)(long)arg;
    for (int i = 0; i < PER_PROD; i++) {
        int16_t v[12];
        for (int k = 0; k < 12; k++) v[k] = (int16_t)(id * 1000 + (i % 1000) + k);
        tars_log_ring_push(&g_ring, i, TLE_IMU, (uint8_t)id, v, 12);
    }
    return NULL;
}
static void *consumer(void *arg)
{
    (void)arg;
    uint32_t tail = 0;
    long good = 0, torn = 0, lost = 0, bad = 0;
    while (!g_done || tail < tars_log_ring_head(&g_ring)) {
        uint32_t head = tars_log_ring_head(&g_ring);
        uint32_t first = tars_log_ring_first(&g_ring);
        if (tail < first) {
            lost += first - tail;
            tail = first;
        }
        if (tail >= head) continue;
        tars_log_ev_t e;
        int r = tars_log_ring_get(&g_ring, tail, &e);
        if (r == 0) continue;
        tail++;
        if (r < 0) {
            torn++;
            continue;
        }
        /* every committed event must be internally consistent */
        const int id = e.id;
        const int base = e.v[0];
        int ok = (id >= 0 && id < NPROD) && base >= id * 1000 && base < id * 1000 + 1000;
        for (int k = 0; k < 12 && ok; k++) ok = e.v[k] == (int16_t)(base + k);
        if (ok) good++; else bad++;
    }
    printf("   concurrency: %ld good, %ld torn/overwritten, %ld lost by lapping, %ld inconsistent\n", good, torn, lost, bad);
    CHECK(bad == 0);
    CHECK(good > 0);
    CHECK(good + torn + lost + bad == (long)NPROD * PER_PROD);      /* every index is accounted for exactly once */
    return NULL;
}
static void test_ring_threads(void)
{
    tars_log_ev_t *mem = calloc(512, sizeof *mem);
    tars_log_ring_init(&g_ring, mem, 512);
    tars_log_ring_reset(&g_ring, true, 0);
    pthread_t p[NPROD], c;
    pthread_create(&c, NULL, consumer, NULL);
    for (long i = 0; i < NPROD; i++) pthread_create(&p[i], NULL, producer, (void *)i);
    for (int i = 0; i < NPROD; i++) pthread_join(p[i], NULL);
    g_done = 1;
    pthread_join(c, NULL);
    CHECK(tars_log_ring_head(&g_ring) == (uint32_t)(NPROD * PER_PROD));
    free(mem);
}

static void test_waves(void)
{
    tars_log_wave_t w;
    memset(&w, 0, sizeof w);
    w.kind = TLW_STEP; w.id = 2; w.amp = 40; w.hold_ms = 600; w.lead_ms = 400; w.tail_ms = 300;
    CHECK(tars_log_wave_duration_ms(&w) == 400 + 4 * 600 + 300);
    float p[3];
    CHECK(tars_log_wave_eval(&w, 0, p) && p[0] == 0 && p[1] == 0 && p[2] == 0);
    CHECK(tars_log_wave_eval(&w, 399, p) && p[1] == 0);
    CHECK(tars_log_wave_eval(&w, 400, p) && p[1] == 40 && p[0] == 0 && p[2] == 0);
    CHECK(tars_log_wave_eval(&w, 999, p) && p[1] == 40);
    CHECK(tars_log_wave_eval(&w, 1000, p) && p[1] == 0);
    CHECK(tars_log_wave_eval(&w, 1600, p) && p[1] == -40);
    CHECK(tars_log_wave_eval(&w, 2200, p) && p[1] == 0);
    CHECK(tars_log_wave_eval(&w, 2799, p) && p[1] == 0);
    CHECK(!tars_log_wave_eval(&w, 3100, p) && p[1] == 0);
    w.id = 4;
    CHECK(!tars_log_wave_eval(&w, 500, p));

    memset(&w, 0, sizeof w);
    w.kind = TLW_SINE; w.id = 1; w.amp = 30; w.freq_mhz = 2000; w.cycles = 4; w.lead_ms = 400; w.tail_ms = 300;
    CHECK(tars_log_wave_duration_ms(&w) == 400 + 4 * 500 + 300);
    CHECK(tars_log_wave_eval(&w, 400, p) && fabsf(p[0]) < 1e-3f);
    CHECK(tars_log_wave_eval(&w, 400 + 125, p) && fabsf(p[0] - 30.0f) < 0.05f);      /* quarter period of 2 Hz */
    CHECK(tars_log_wave_eval(&w, 400 + 375, p) && fabsf(p[0] + 30.0f) < 0.05f);
    CHECK(tars_log_wave_eval(&w, 400 + 2000 + 10, p) && p[0] == 0);                  /* after the last cycle */
    CHECK(!tars_log_wave_eval(&w, 2800, p));
    float peak = 0;
    for (int t = 400; t < 2400; t++) {
        tars_log_wave_eval(&w, t, p);
        if (fabsf(p[0]) > peak) peak = fabsf(p[0]);
    }
    CHECK(fabsf(peak - 30.0f) < 0.01f);
}

static void test_filter(void)
{
    tars_log_cmd_filter_t f;
    memset(&f, 0, sizeof f);
    int16_t a[3] = {430, 540, 492};
    CHECK(tars_log_cmd_should_log(&f, 0, a));              /* first one */
    CHECK(!tars_log_cmd_should_log(&f, 2000, a));          /* unchanged, 2 ms */
    CHECK(!tars_log_cmd_should_log(&f, 5000, a));
    CHECK(tars_log_cmd_should_log(&f, 12000, a));          /* 10 ms since last logged */
    int16_t b[3] = {430, 560, 492};
    CHECK(!tars_log_cmd_should_log(&f, 12500, b));         /* big move but only 0.5 ms later */
    CHECK(tars_log_cmd_should_log(&f, 13500, b));          /* 1.5 ms and moved 20 counts */
    int16_t c[3] = {430, 563, 492};
    CHECK(!tars_log_cmd_should_log(&f, 16000, c));         /* moved only 3 */
}

int main(void)
{
    test_numbers();
    test_lines();
    test_ring_basic();
    test_ring_threads();
    test_waves();
    test_filter();
    printf("test_core: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
