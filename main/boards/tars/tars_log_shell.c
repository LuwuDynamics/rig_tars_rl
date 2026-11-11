/*
 * tars_log_shell.c - command shell, recording sessions and servo test driver of the Tars log tool.
 *
 * Portable C99 (no ESP-IDF): everything it needs from the robot goes through tars_log_port.h, so the same code is
 * exercised on the host by tools/tars_log/test/ against a simulated robot and the real PC tool (tars_serial.py).
 *
 * Protocol (device -> host), every line is one of
 *   "# text"                       human readable information
 *   "@ok <cmd> ..." / "@err ..."   command result
 *   "@begin|@meta|@fmt|@end ..."   framing of a recording dump
 *   "@I|S|C|V|M,<t_us>,...*HH"     data events, HH = XOR checksum of the text between '@' and '*'
 * Commands (host -> device) are single text lines, see cmd_help().
 */
#include "tars_log_shell.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "tars_log_core.h"
#include "tars_log_port.h"

#define FW_TAG "tars-log/1"

/* ---- safety limits for everything that moves a servo (servo counts, WritePos units: 1024 counts = 300 deg) ---- */
#define HIP_MAX_COUNTS   120      /* +-35 deg  (the classic gait uses +-35 counts) */
#define LIFT_MAX_COUNTS  110      /* about +-14 mm of leg lift (= the classic gait maximum) */
#define POSE_DEFAULT_S   10
#define POSE_MAX_S       120
#define LIMP_MAX_S       600
#define REC_MAX_S        600
#define WALK_MAX_VX      35
#define TURN_MAX_VYAW    25

/* ---- servo read-back schedule (one bus request per robot cycle, which is ~3 ms) ---- */
#define FB_PERIOD_US     20000    /* every servo 50 Hz while a session is running */
#define FB_FOCUS_US       5000    /* the servo under test 200 Hz ... */
#define FB_OTHER_US      25000    /* ... the others 40 Hz */
#define FB_SLOW_US      100000    /* only holding a pose / limp: 10 Hz */
#define VOLT_FAST_US    100000      /* 10 Hz while recording / watching: the supply sag during walking is what we are after */
#define VOLT_SLOW_US  20000000
#define POSE_RESEND_US   50000
#define LIMP_SEND_US     30000
#define WAVE_SETTLE_US  200000

#define LINE_MAX_CHARS   160
#define OBUF_CHARS       288
#define DBUF_CHARS      1280

typedef enum { MODE_IDLE = 0, MODE_WATCH, MODE_REC, MODE_LIVE, MODE_SCAN, MODE_DUMP } sh_mode_t;
typedef enum { OWN_NONE = 0, OWN_WAVE, OWN_POSE, OWN_LIMP, OWN_RESTORE } sh_own_t;
enum { ACT_NONE = 0, ACT_WALK, ACT_TURN };

typedef struct {
    int16_t pos, spd, load;
    int64_t t_us;       /* when the last state frame arrived (0 = never) */
    int64_t req_us;     /* when the matching read request was sent */
} sh_fb_t;

typedef struct {
    char b[OBUF_CHARS];
    size_t n;
} sh_line_t;

static struct {
    /* ring */
    tars_log_ring_t ring;
    bool ring_ok;
    volatile bool rec_on;
    tars_log_cmd_filter_t cmd_filter;
    /* session */
    sh_mode_t mode;
    int64_t t0_us;
    int64_t deadline_us;            /* 0 = none */
    char kind[12];
    char args[72];
    int act;
    bool is_wave;
    bool imu_on;
    /* dump */
    int dump_phase;                 /* 0 header, 1 events, 2 trailer, 3 done */
    bool dump_aborted_msg;
    uint32_t dump_idx, dump_end, dump_skipped;
    int64_t dump_not_before_us;
    size_t dlen, doff;              /* bytes waiting in dbuf / already sent */
    int64_t dstall_us;              /* since when the host has not been accepting data (0 = flowing) */
    /* live */
    uint32_t live_tail, live_lost;
    /* watch */
    int64_t watch_next_us;
    int32_t watch_period_ms;
    /* input */
    char line[LINE_MAX_CHARS];
    size_t line_n;
    bool line_over;
    /* scratch (log task only) */
    char obuf[OBUF_CHARS];
    char dbuf[DBUF_CHARS];
    char lbuf[160];
    /* latest samples (written by the data hooks, read by watch / status / scan) */
    tars_log_imu_t imu;
    int64_t imu_t_us;
    sh_fb_t fb[3];
    int16_t cmd[3];
    int16_t cmd_speed;
    int64_t cmd_t_us;
    int16_t volt_cv;
    int64_t volt_t_us;
    /* servo ownership and read-back scheduling: shared between the log task and the robot task, guarded by tlp_lock() */
    volatile int own;
    volatile bool bus_fast;
    volatile int focus;
    volatile bool sched_reset;
    tars_log_wave_t wave;
    int64_t wave_t0_us;
    bool wave_started;
    volatile bool wave_done;
    int64_t wave_done_us;
    float pose[3];
    int64_t pose_deadline_us, pose_sent_us;
    int64_t limp_deadline_us, limp_sent_us;
    int limp_next, restore_n;
    int64_t next_due[3];
    int64_t volt_due_us;
    /* learned-mode dead-man timer: 'learn on go <sec>' switches itself off after <sec> */
    int64_t learn_deadline_us;
    int plain_move_active;                    /* 'walk' / 'turn' (no recording) was started: 'stop' ends it too */
    int64_t walk_start_us;                    /* 'test walk|turn ... <delay>': the walk starts at this time (the recording is already running) */
    int walk_val, walk_ms, walk_turn;
} S;

/* recording flag: the release/acquire pair guarantees that a producer that sees `true` also sees the freshly reset ring */
static inline bool rec_get(void) { return __atomic_load_n(&S.rec_on, __ATOMIC_ACQUIRE); }
static inline void rec_set(bool on) { __atomic_store_n(&S.rec_on, on, __ATOMIC_RELEASE); }

/* ---------------------------------------------------------------------------------------------------------------
 * output helpers (log task only)
 * --------------------------------------------------------------------------------------------------------------- */
static void ln_reset(sh_line_t *l) { l->n = 0; l->b[0] = 0; }
static void ln_s(sh_line_t *l, const char *s)
{
    while (*s && l->n < sizeof l->b - 2) l->b[l->n++] = *s++;
    l->b[l->n] = 0;
}
static void ln_i(sh_line_t *l, int32_t v)
{
    if (l->n + 14 < sizeof l->b) {
        l->n += tars_log_put_i32(l->b + l->n, v);
        l->b[l->n] = 0;
    }
}
static void ln_m(sh_line_t *l, int32_t milli, int dec)
{
    if (l->n + 16 < sizeof l->b) {
        l->n += tars_log_put_milli(l->b + l->n, milli, dec);
        l->b[l->n] = 0;
    }
}
static void ln_send(sh_line_t *l)
{
    if (l->n < sizeof l->b - 1) l->b[l->n++] = '\n';
    tlp_out(l->b, l->n);
    ln_reset(l);
}

static void outf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void outf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(S.obuf, sizeof S.obuf - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n > sizeof S.obuf - 2) n = (int)sizeof S.obuf - 2;
    S.obuf[n++] = '\n';
    tlp_out(S.obuf, (size_t)n);
}

void tars_log_shell_say(const char *text)
{
    outf("# %s", text);
}

static int32_t div_round(int32_t a, int32_t b)
{
    return (a >= 0 ? a + b / 2 : a - b / 2) / b;
}
static int32_t acc_milli(int16_t lsb) { return div_round((int32_t)lsb * 9800, 16384); }   /* m/s^2 * 1000 */
static int32_t gyro_milli(int16_t lsb) { return div_round((int32_t)lsb * 125, 4); }       /* deg/s * 1000 */

static const char *mode_name(void)
{
    switch (S.mode) {
    case MODE_WATCH: return "watch";
    case MODE_REC: return S.kind;
    case MODE_LIVE: return "live";
    case MODE_SCAN: return "scan";
    case MODE_DUMP: return "dump";
    default: return "idle";
    }
}

static const char *own_name(void)
{
    switch (S.own) {
    case OWN_WAVE: return "test";
    case OWN_POSE: return "pose";
    case OWN_LIMP: return "limp";
    case OWN_RESTORE: return "restoring";
    default: return "none";
    }
}

/* ---------------------------------------------------------------------------------------------------------------
 * resources and sessions
 * --------------------------------------------------------------------------------------------------------------- */
static bool ensure_ring(void)
{
    if (S.ring_ok) return true;
    uint32_t cap = 0;
    tars_log_ev_t *mem = tlp_alloc_events(&cap);
    if (!mem || cap < 256) return false;
    tars_log_ring_init(&S.ring, mem, cap);
    S.ring_ok = true;
    return true;
}

static void bus_fast_set(bool on, int focus)
{
    tlp_lock();
    S.bus_fast = on;
    S.focus = on ? focus : 0;
    S.sched_reset = true;
    tlp_unlock();
}

static void imu_set(bool on)
{
    if (S.imu_on == on) return;
    S.imu_on = on;
    tlp_imu_stream(on);
}

static void release_servos(void)
{
    tlp_lock();
    if (S.own == OWN_LIMP) {
        S.own = OWN_RESTORE;
        S.restore_n = 0;
    } else if (S.own != OWN_NONE && S.own != OWN_RESTORE) {
        S.own = OWN_NONE;
    }
    tlp_unlock();
}

static bool servo_use_ok(void)
{
    if (tlp_calibrating()) {
        outf("@err robot is in servo calibration mode (triple-click BOOT to leave it)");
        return false;
    }
    return true;
}

static bool idle_or_busy(const char *cmd)
{
    if (S.mode == MODE_IDLE) return true;
    outf("@err %s: busy (%s) - send 'stop' first", cmd, mode_name());
    return false;
}

static void push_snapshot(int64_t now)
{
    /* first events of a recording: the last known servo / command / voltage state, so every trace starts with context */
    for (int i = 0; i < 3; i++) {
        if (S.fb[i].t_us) {
            int16_t v[4] = {S.fb[i].pos, S.fb[i].spd, S.fb[i].load, 0};
            tars_log_ring_push(&S.ring, now, TLE_SERVO, (uint8_t)(i + 1), v, 4);
        }
    }
    if (S.cmd_t_us) {
        int16_t v[4] = {S.cmd[0], S.cmd[1], S.cmd[2], S.cmd_speed};
        tars_log_ring_push(&S.ring, now, TLE_CMD, 0, v, 4);
        S.cmd_filter.have = true;
        S.cmd_filter.last_us = now;
        for (int i = 0; i < 3; i++) S.cmd_filter.last[i] = S.cmd[i];
    }
    if (S.volt_t_us) {
        int16_t v[1] = {S.volt_cv};
        tars_log_ring_push(&S.ring, now, TLE_VOLT, 0, v, 1);
    }
}

static void push_text(int64_t now, const char *text)
{
    int16_t v[TARS_LOG_EV_VALS];
    memset(v, 0, sizeof v);
    size_t n = strlen(text);
    if (n > TARS_LOG_MSG_CHARS) n = TARS_LOG_MSG_CHARS;
    memcpy(v, text, n);
    tars_log_ring_push(&S.ring, now, TLE_MSG, 0, v, TARS_LOG_EV_VALS);
}

static bool begin_record(const char *kind, const char *args, int32_t sec, int act, bool wave, int focus)
{
    if (!ensure_ring()) {
        outf("@err no memory for the log buffer");
        return false;
    }
    const int64_t now = tlp_now_us();
    rec_set(false);
    tars_log_ring_reset(&S.ring, false, now);
    strncpy(S.kind, kind, sizeof S.kind - 1);
    S.kind[sizeof S.kind - 1] = 0;
    strncpy(S.args, args, sizeof S.args - 1);
    S.args[sizeof S.args - 1] = 0;
    S.act = act;
    S.is_wave = wave;
    S.t0_us = now;
    S.deadline_us = sec > 0 ? now + (int64_t)sec * 1000000 : 0;
    S.mode = MODE_REC;
    bus_fast_set(true, focus);
    imu_set(true);
    push_snapshot(now);
    push_text(now, kind);
    rec_set(true);
    return true;
}

static void dump_start(void);

static void finish_recording(const char *why)
{
    if (S.mode != MODE_REC) return;
    S.walk_start_us = 0;                                        /* a delayed walk that has not started yet is cancelled */
    rec_set(false);
    bus_fast_set(false, 0);
    imu_set(false);
    if (S.act != ACT_NONE) tlp_gait_stop();
    if (S.is_wave) {
        tlp_lock();
        if (S.own == OWN_WAVE) S.own = OWN_NONE;
        tlp_unlock();
    }
    const uint32_t events = tars_log_ring_head(&S.ring);
    outf("# recording finished (%s): %u events, %u dropped", why, (unsigned)events, (unsigned)S.ring.dropped);
    dump_start();
    S.dump_end = events;
    S.dump_not_before_us = tlp_now_us() + 30000;   /* let a producer that was mid-push commit its slot */
}

static void stop_resources(void)
{
    rec_set(false);
    bus_fast_set(false, 0);
    imu_set(false);
}

/* ---------------------------------------------------------------------------------------------------------------
 * dump: text is collected in dbuf and handed to the link without ever dropping or repeating a byte; if the host stops
 * reading the dump waits (up to DUMP_STALL_US), then gives up - the recording stays in RAM and 'dump' sends it again.
 * --------------------------------------------------------------------------------------------------------------- */
#define DUMP_STALL_US 8000000

static void dbuf_line(const char *s)
{
    const size_t n = strlen(s);
    if (S.dlen + n + 1 <= sizeof S.dbuf) {
        memcpy(S.dbuf + S.dlen, s, n);
        S.dlen += n;
        S.dbuf[S.dlen++] = '\n';
    }
}

static void dbuf_linef(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void dbuf_linef(const char *fmt, ...)
{
    char tmp[200];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    dbuf_line(tmp);
}

/* try to send what waits in dbuf; true = empty now. Gives up (aborts the dump) after DUMP_STALL_US of silence from the host. */
static bool dump_drain(int64_t now)
{
    while (S.doff < S.dlen) {
        const size_t w = tlp_write(S.dbuf + S.doff, S.dlen - S.doff);
        S.doff += w;
        if (S.doff < S.dlen) break;
    }
    if (S.doff >= S.dlen) {
        S.dlen = S.doff = 0;
        S.dstall_us = 0;
        return true;
    }
    if (!S.dstall_us) S.dstall_us = now;
    if (now - S.dstall_us > DUMP_STALL_US) {
        S.dlen = S.doff = 0;
        S.dstall_us = 0;
        S.mode = MODE_IDLE;
        S.dump_aborted_msg = true;
    }
    return false;
}

static void dump_start(void)
{
    S.mode = MODE_DUMP;
    S.dump_phase = 0;
    S.dlen = S.doff = 0;
    S.dstall_us = 0;
    S.dump_idx = tars_log_ring_first(&S.ring);
    S.dump_end = tars_log_ring_head(&S.ring);
    S.dump_skipped = 0;
}

static void dump_header(void)
{
    dbuf_linef("@begin dump kind=%s events=%u dropped=%u", S.kind, (unsigned)(S.dump_end - S.dump_idx), (unsigned)S.ring.dropped);
    dbuf_linef("@meta fw=" FW_TAG " zero=%d,%d,%d mask=%d,%d,%d acc_lsb_per_g=16384 g_mm_s2=9800 gyro_lsb_per_dps=32 up_scale=10000 vxy=int8x2",
               tlp_zero_pos(0), tlp_zero_pos(1), tlp_zero_pos(2), tlp_servo_mask(0), tlp_servo_mask(1), tlp_servo_mask(2));
    dbuf_linef("@meta args=%s", S.args[0] ? S.args : "-");
    dbuf_line("@fmt I t_us,ax,ay,az,gx,gy,gz,ux,uy,uz,flags,state,vxy");
    dbuf_line("@fmt S t_us,id,pos,spd,load,lat_us");
    dbuf_line("@fmt C t_us,src,p1,p2,p3,speed");
    dbuf_line("@fmt V t_us,centivolt");
    dbuf_line("@fmt M t_us,text");
}

static bool dump_step(int64_t now)
{
    if (now < S.dump_not_before_us) return true;
    if (S.dlen && !dump_drain(now)) return S.mode == MODE_DUMP;
    switch (S.dump_phase) {
    case 0:
        dump_header();
        S.dump_phase = 1;
        break;
    case 1: {
        int budget = 48;
        while (budget-- > 0 && S.dump_idx < S.dump_end && sizeof S.dbuf - S.dlen >= 150) {
            tars_log_ev_t e;
            const int r = tars_log_ring_get(&S.ring, S.dump_idx, &e);
            S.dump_idx++;
            if (r != 1) {
                S.dump_skipped++;
                continue;
            }
            const size_t len = tars_log_format_event(&e, S.dbuf + S.dlen, sizeof S.dbuf - S.dlen);
            if (!len) S.dump_skipped++;
            S.dlen += len;
        }
        if (S.dump_idx >= S.dump_end) S.dump_phase = 2;
        break;
    }
    case 2:
        dbuf_linef("@end dump events=%u skipped=%u dropped=%u", (unsigned)(S.dump_end - tars_log_ring_first(&S.ring) - S.dump_skipped),
                   (unsigned)S.dump_skipped, (unsigned)S.ring.dropped);
        dbuf_linef("@ok %s done", S.kind);
        S.dump_phase = 3;
        break;
    default:
        S.mode = MODE_IDLE;
        return false;
    }
    dump_drain(now);
    if (S.mode == MODE_DUMP && S.dump_phase == 3 && S.dlen == 0) {      /* the trailer is out: the session is over right now, not one poll later */
        S.mode = MODE_IDLE;
        return false;
    }
    return S.mode == MODE_DUMP;
}

/* ---------------------------------------------------------------------------------------------------------------
 * live stream
 * --------------------------------------------------------------------------------------------------------------- */
static bool live_step(void)
{
    const uint32_t head = tars_log_ring_head(&S.ring);
    const uint32_t first = tars_log_ring_first(&S.ring);
    if (S.live_tail < first) {
        S.live_lost += first - S.live_tail;
        S.live_tail = first;
    }
    size_t n = 0;
    int budget = 48;
    while (budget-- > 0 && S.live_tail < head) {
        tars_log_ev_t e;
        const int r = tars_log_ring_get(&S.ring, S.live_tail, &e);
        if (r == 0) break;
        S.live_tail++;
        if (r < 0) {
            S.live_lost++;
            continue;
        }
        const size_t len = tars_log_format_event(&e, S.lbuf, sizeof S.lbuf);
        if (!len) continue;
        if (n + len > sizeof S.dbuf) {
            if (!tlp_out(S.dbuf, n)) S.live_lost += 1;
            n = 0;
        }
        memcpy(S.dbuf + n, S.lbuf, len);
        n += len;
    }
    if (n && !tlp_out(S.dbuf, n)) S.live_lost += 1;
    const uint32_t backlog = head - S.live_tail;
    if (backlog > S.ring.cap / 2) {                      /* the link cannot keep up: skip ahead instead of lagging forever */
        S.live_lost += backlog - S.ring.cap / 4;
        S.live_tail = head - S.ring.cap / 4;
    }
    return head > S.live_tail;
}

/* ---------------------------------------------------------------------------------------------------------------
 * watch / scan / status
 * --------------------------------------------------------------------------------------------------------------- */
static int32_t age_ms(int64_t t_us, int64_t now)
{
    if (!t_us) return -1;
    int64_t a = (now - t_us) / 1000;
    return a < 0 ? 0 : (a > 9999 ? 9999 : (int32_t)a);
}

static int32_t servo_off(int i)
{
    return tlp_servo_mask(i) * ((int32_t)S.fb[i].pos - (int32_t)tlp_zero_pos(i));
}

static void print_watch(int64_t now)
{
    sh_line_t l;
    ln_reset(&l);
    ln_s(&l, "# w t=");
    ln_m(&l, (int32_t)((now - S.t0_us) / 1000), 2);
    ln_s(&l, " acc=");
    for (int i = 0; i < 3; i++) {
        ln_m(&l, acc_milli(S.imu.acc[i]), 2);
        if (i < 2) ln_s(&l, ",");
    }
    {
        const float ax = S.imu.acc[0], ay = S.imu.acc[1], az = S.imu.acc[2];
        ln_s(&l, " |a|=");
        ln_m(&l, (int32_t)(sqrtf(ax * ax + ay * ay + az * az) * (9800.0f / 16384.0f) + 0.5f), 2);
    }
    ln_s(&l, " gyro=");
    for (int i = 0; i < 3; i++) {
        ln_m(&l, gyro_milli(S.imu.gyro[i]), 1);
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " up=");
    for (int i = 0; i < 3; i++) {
        ln_m(&l, div_round(S.imu.up[i], 10), 2);
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " pos=");
    for (int i = 0; i < 3; i++) {
        ln_i(&l, S.fb[i].pos);
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " off=");
    for (int i = 0; i < 3; i++) {
        ln_i(&l, servo_off(i));
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " load=");
    for (int i = 0; i < 3; i++) {
        ln_i(&l, S.fb[i].load);
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " age=");
    for (int i = 0; i < 3; i++) {
        ln_i(&l, age_ms(S.fb[i].t_us, now));
        if (i < 2) ln_s(&l, ",");
    }
    ln_s(&l, " V=");
    ln_m(&l, S.volt_cv * 10, 2);
    ln_s(&l, " cmd=");
    for (int i = 0; i < 3; i++) {
        ln_i(&l, S.cmd[i]);
        if (i < 2) ln_s(&l, ",");
    }
    if (S.imu.state >= 0) {                                /* learned mode on: supervisor state (keep in sync with tars_state_t in tars_ctrl.h) */
        static const char *const kState[] = {"STAND", "WALK_F", "WALK_B", "TURN_L", "TURN_R", "FALLEN", "GETUP1", "GETUP2A", "GETUP2B", "GETUP2C", "FAILED", "GETUP_NN", "MOTION"};
        ln_s(&l, " st=");
        ln_s(&l, S.imu.state < (int)(sizeof kState / sizeof kState[0]) ? kState[S.imu.state] : "?");
    }
    ln_send(&l);
}

static void finish_scan(void)
{
    stop_resources();
    S.mode = MODE_IDLE;
    for (int i = 0; i < 3; i++) {
        if (S.fb[i].t_us >= S.t0_us) {
            outf("# servo %d: pos=%d off=%d spd=%d load=%d", i + 1, S.fb[i].pos, (int)servo_off(i), S.fb[i].spd, S.fb[i].load);
        } else {
            outf("# servo %d: NO REPLY (check cable / ID / power)", i + 1);
        }
    }
    if (S.volt_t_us >= S.t0_us) {
        sh_line_t l;
        ln_reset(&l);
        ln_s(&l, "# servo bus voltage: ");
        ln_m(&l, S.volt_cv * 10, 2);
        ln_s(&l, " V");
        ln_send(&l);
    } else {
        outf("# servo bus voltage: no reply");
    }
    outf("@ok scan");
}

/* ---------------------------------------------------------------------------------------------------------------
 * commands
 * --------------------------------------------------------------------------------------------------------------- */
static void cmd_help(int argc, char **argv)
{
    static const char *const kHelp[] = {
        "tars-log commands (one line each, '@ok'/'@err' answers):",
        "  hello | info | status           identify / one-screen state",
        "  scan                            ping the 3 servos once and read the bus voltage",
        "  watch [hz]                      print a live status line (default 5 Hz); 'stop' ends it",
        "  rec <sec>                       record IMU + servos for <sec> s (0 = until stop), dump when done",
        "  live                            stream raw events in real time until 'stop'",
        "  stop                            end whatever is running (also releases held servos)",
        "  estop                           EMERGENCY stop (= Ctrl-C): also stops walking and the learned mode, legs go to the zero pose",
        "  dump | clear                    send / discard the last recording",
        "  mark <text>                     put a marker into the recording",
        "  test hold <sec>                 robot still: sensor noise / bias / voltage baseline",
        "  test step <id> <amp> [hold_ms]  servo step response  (id 1..3, amp counts, e.g. 'test step 1 40')",
        "  test sine <id> <amp> <hz> [n]   servo sine tracking  (e.g. 'test sine 1 30 1.5 5')",
        "  test walk <vx> <sec> [delay]    forward walk (classic, or the learned gait when it is on) while recording; [delay] s: walk later (unplug USB meanwhile)",
        "  test turn <vyaw> <sec> [delay]  turn (counts -25..25) while recording; [delay] as for walk",
        "  pose <c1> <c2> <c3> [sec]       hold servos at offsets from the zero pose (counts)",
        "  release                         give the servos back to the normal gait",
        "  torque 0|1                      0 = servos limp (turn legs by hand, use 'watch'), 1 = back on",
        "  cal 0|1                         IMU axis calibration (0 upright & still, 1 nose-down ~30 deg & still)",
        "  walk <vx> <sec> / turn <vyaw> <sec>  move for <sec> s WITHOUT a recording (use with 'watch': the data streams live, survives a power loss)",
        "  learn dry|on go [sec]|off       learned behaviour (dry = compute only; 'on go' switches itself off after [sec], default 30)",
        "  learn auto off|on|walk go       learned mode at power-up (stored; default walk when nothing is stored): on = stand + get up by itself, walk = + walking commands",
        "  learn log                       what the learned mode did (state changes with times; RAM, lost at power-off)",
        "  errors                          last 10 firmware errors saved in flash",
        "  learn gait [0..6] [save]          walking gait set (default 6): 0 = v4 (original simulator), 1 = v5 (real-calibrated simulator), 2 = v5 + faster forward, 3 = v6 (10 mm foot plates), 4 = v7 (smoother forward), 5 = v8 (v7 + smooth backward), 6 = v9 (sinusoid hips); 'save' keeps it over power cycles",
        "  learn gait custom <f|b|t|r> <15 ints>  replace ONE gait of the set by parameters (x 1e6, order q0 a1 b1 a2 b2 z0 c1 d1 c2 d2 f kp_h kd_h kp_z kd_z); volatile, 'learn gait N' undoes it",
        "  learn straight [on|off|kp kd [trim]] forward / backward walks keep their heading (gyro heading hold: kp %/rad 0..300 default 150, kd 0..100 default 0, trim -300..300 per mille, + = turn right); ON at power-up, volatile",
        "  learn motion [<name> [speed%]|stop]  show motions, gestures and emotions (the robot MOVES where it stands, ~5..12 s, then stands again; a gesture first retracts the arms and stands on the torso base): no argument lists them; speed 30..100 % (default 100); stop ends one; needs the learned mode ON",
        "  learn getup [kf|alt|nn] [save]    get-up after a fall: kf = keyframes, alt = learned network on the 1st and 3rd attempt, keyframes on the 2nd (default), nn = network only; 'save' keeps it over power cycles",
        "  learn speed [on|off|max 70..100] walk / turn command magnitude (counts) scales the gait amplitude: <=14 -> 70 %, >=20 -> max (default 90 %); off = always the full gait",
        "  learn tune [kp% kd% lpf_hz amp% hz]  walking feedback tuning (volatile): pitch fb, rate fb, rate low-pass, gait amplitude, controller rate 50|100|250|500; default 100 100 0 100 500",
        "  map <flip 0|1> <swap 0|1>       servo direction hypothesis (stored); or  map <s1> <s2> <s3> <swap>  with s = +1/-1 per servo",
        "  selftest 0..3                   servo direction self-test poses",
        "  imu [regs|raw [n]|reinit|fs <0..3>]  QMI8658 register diagnostics (reinit / fs change the chip until reboot)",
        "  link usb|console                answer path: direct (default) or the plain console (slow fallback)",
        "  baud <rate>                     UART speed, e.g. 921600 (PC tool: --fast); reverts to 115200 if the PC does not follow",
        "  reboot",
    };
    (void)argc;
    (void)argv;
    for (size_t i = 0; i < sizeof kHelp / sizeof kHelp[0]; i++) outf("# %s", kHelp[i]);
    outf("@ok help");
}

static void cmd_hello(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    char info[200];
    info[0] = 0;
    tlp_info(info, sizeof info);
    outf("@ok hello fw=" FW_TAG " %s", info);
}

static void cmd_status(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    const int64_t now = tlp_now_us();
    outf("# mode=%s servos=%s bus=%s imu_stream=%d recording=%d events=%u/%u dropped=%u calibrating=%d", mode_name(), own_name(),
         S.bus_fast ? "fast" : "slow", S.imu_on ? 1 : 0, rec_get() ? 1 : 0, (unsigned)(S.ring_ok ? tars_log_ring_head(&S.ring) : 0u),
         (unsigned)(S.ring_ok ? S.ring.cap : 0u), (unsigned)(S.ring_ok ? S.ring.dropped : 0u), tlp_calibrating());
    for (int i = 0; i < 3; i++) {
        if (!S.fb[i].t_us) {
            outf("# servo %d: no read-back yet (starts with scan / watch / rec)", i + 1);
            continue;
        }
        outf("# servo %d: pos=%d off=%d spd=%d load=%d age_ms=%d", i + 1, S.fb[i].pos, (int)servo_off(i), S.fb[i].spd, S.fb[i].load,
             (int)age_ms(S.fb[i].t_us, now));
    }
    char rl[220];
    rl[0] = 0;
    tlp_rl_status(rl, sizeof rl);
    outf("# rl: %s", rl);
    outf("@ok status");
}

static void cmd_scan(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    if (!idle_or_busy("scan")) return;
    const int64_t now = tlp_now_us();
    S.mode = MODE_SCAN;
    S.t0_us = now;
    S.deadline_us = now + 350000;
    bus_fast_set(true, 0);
}

static void cmd_watch(int argc, char **argv)
{
    if (!idle_or_busy("watch")) return;
    int32_t hz = 5;
    if (argc > 1 && (!tars_log_parse_i32(argv[1], &hz) || hz < 1 || hz > 20)) {
        outf("@err usage: watch [hz 1..20]");
        return;
    }
    const int64_t now = tlp_now_us();
    S.t0_us = now;
    S.watch_period_ms = 1000 / hz;
    S.watch_next_us = now;
    S.mode = MODE_WATCH;
    bus_fast_set(true, 0);
    imu_set(true);
    outf("@ok watch %d Hz (send 'stop' to end)", (int)hz);
}

static void cmd_rec(int argc, char **argv)
{
    if (!idle_or_busy("rec")) return;
    int32_t sec = 0;
    if (argc < 2 || !tars_log_parse_i32(argv[1], &sec) || sec < 0 || sec > REC_MAX_S) {
        outf("@err usage: rec <seconds 0..%d>  (0 = until 'stop')", REC_MAX_S);
        return;
    }
    if (begin_record("rec", "", sec, ACT_NONE, false, 0)) {
        outf("@ok rec started %d s, capacity %u events", (int)sec, (unsigned)S.ring.cap);
    }
}

static void cmd_live(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    if (!idle_or_busy("live")) return;
    if (!ensure_ring()) {
        outf("@err no memory for the log buffer");
        return;
    }
    const int64_t now = tlp_now_us();
    rec_set(false);
    tars_log_ring_reset(&S.ring, true, now);
    S.live_tail = 0;
    S.live_lost = 0;
    S.t0_us = now;
    strcpy(S.kind, "live");
    S.args[0] = 0;
    S.mode = MODE_LIVE;
    bus_fast_set(true, 0);
    imu_set(true);
    outf("@ok live started");
    outf("@fmt I t_us,ax,ay,az,gx,gy,gz,ux,uy,uz,flags,state,vxy");
    outf("@fmt S t_us,id,pos,spd,load,lat_us");
    outf("@fmt C t_us,src,p1,p2,p3,speed");
    outf("@fmt V t_us,centivolt");
    outf("@fmt M t_us,text");
    push_snapshot(now);
    rec_set(true);
}

static void stop_everything(void)
{
    switch (S.mode) {
    case MODE_WATCH:
        stop_resources();
        S.mode = MODE_IDLE;
        break;
    case MODE_SCAN:
        stop_resources();
        S.mode = MODE_IDLE;
        break;
    case MODE_LIVE:
        stop_resources();
        S.mode = MODE_IDLE;
        outf("@end live events=%u lost=%u", (unsigned)S.live_tail, (unsigned)S.live_lost);
        break;
    case MODE_REC:
        finish_recording("stopped");
        break;
    case MODE_DUMP:
        S.mode = MODE_IDLE;
        S.dlen = S.doff = 0;
        outf("# dump aborted");
        break;
    default:
        break;
    }
    if (S.plain_move_active) {                                  /* a 'walk' / 'turn' without recording is still running: stop it with the session */
        S.plain_move_active = 0;
        tlp_gait_stop();
    }
    release_servos();
}

static void cmd_stop(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    stop_everything();
    outf("@ok stop");
}

/* emergency stop (also what Ctrl-C sends): walking stops, the learned mode is switched off (legs go to the zero pose), servos are
 * released, sessions end.  Recordings that were running are still dumped. */
static void cmd_estop(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    tlp_gait_stop();
    S.learn_deadline_us = 0;
    if (tlp_rl_active()) tlp_rl_mode(false, true);
    stop_everything();
    tlp_flush_msgs();
    outf("@ok estop");
}

static void cmd_dump(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    if (!idle_or_busy("dump")) return;
    if (!S.ring_ok || tars_log_ring_head(&S.ring) == 0) {
        outf("@err nothing recorded");
        return;
    }
    dump_start();
    S.dump_not_before_us = 0;
}

static void cmd_clear(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    if (!idle_or_busy("clear")) return;
    if (S.ring_ok) tars_log_ring_reset(&S.ring, false, tlp_now_us());
    outf("@ok clear");
}

static void cmd_mark(int argc, char **argv)
{
    char text[TARS_LOG_MSG_CHARS + 1];
    size_t n = 0;
    for (int i = 1; i < argc; i++) {
        for (const char *p = argv[i]; *p && n < TARS_LOG_MSG_CHARS; p++) text[n++] = *p;
        if (i + 1 < argc && n < TARS_LOG_MSG_CHARS) text[n++] = ' ';
    }
    text[n] = 0;
    if (rec_get()) {
        push_text(tlp_now_us(), n ? text : "mark");
        outf("@ok mark");
    } else {
        outf("@ok mark (not recording)");
    }
}

static int servo_limit(int id)
{
    return id == 3 ? LIFT_MAX_COUNTS : HIP_MAX_COUNTS;
}

static bool arm_wave(const tars_log_wave_t *w, const char *kind, const char *args)
{
    if (!servo_use_ok()) return false;
    if (S.own != OWN_NONE) {
        outf("@err servos are held (%s) - send 'release' first", own_name());
        return false;
    }
    const int32_t dur = tars_log_wave_duration_ms(w);
    if (!begin_record(kind, args, (dur + 3000 + 999) / 1000, ACT_NONE, true, w->id)) return false;
    tlp_lock();
    S.wave = *w;
    S.wave_started = false;
    S.wave_done = false;
    S.own = OWN_WAVE;
    tlp_unlock();
    outf("@ok %s started (%d ms) - keep the robot supported, it will move", kind, (int)dur);
    return true;
}

static void cmd_test(int argc, char **argv)
{
    if (!idle_or_busy("test")) return;
    if (argc < 2) {
        outf("@err usage: test hold|step|sine|walk|turn ...");
        return;
    }
    const char *what = argv[1];
    int32_t a = 0, b = 0, c = 0;
    char args[72];
    if (!strcmp(what, "hold")) {
        if (argc < 3 || !tars_log_parse_i32(argv[2], &a) || a < 1 || a > REC_MAX_S) {
            outf("@err usage: test hold <seconds 1..%d>", REC_MAX_S);
            return;
        }
        snprintf(args, sizeof args, "sec=%d", (int)a);
        if (begin_record("hold", args, a, ACT_NONE, false, 0)) outf("@ok hold started %d s", (int)a);
    } else if (!strcmp(what, "step")) {
        if (argc < 4 || !tars_log_parse_i32(argv[2], &a) || !tars_log_parse_i32(argv[3], &b) || a < 1 || a > 3 ||
            (argc > 4 && !tars_log_parse_i32(argv[4], &c))) {
            outf("@err usage: test step <id 1..3> <amp counts> [hold_ms 100..3000]");
            return;
        }
        if (argc <= 4) c = 600;
        if (b == 0 || b > servo_limit((int)a) || b < -servo_limit((int)a) || c < 100 || c > 3000) {
            outf("@err amplitude must be 1..%d counts for servo %d, hold 100..3000 ms", servo_limit((int)a), (int)a);
            return;
        }
        tars_log_wave_t w;
        memset(&w, 0, sizeof w);
        w.kind = TLW_STEP;
        w.id = (uint8_t)a;
        w.amp = (int16_t)b;
        w.hold_ms = (uint16_t)c;
        w.lead_ms = 400;
        w.tail_ms = 300;
        snprintf(args, sizeof args, "id=%d amp=%d hold_ms=%d", (int)a, (int)b, (int)c);
        arm_wave(&w, "step", args);
    } else if (!strcmp(what, "sine")) {
        int32_t f_milli = 0, cycles = 5;
        if (argc < 5 || !tars_log_parse_i32(argv[2], &a) || !tars_log_parse_i32(argv[3], &b) || !tars_log_parse_milli(argv[4], &f_milli) ||
            a < 1 || a > 3 || (argc > 5 && !tars_log_parse_i32(argv[5], &cycles))) {
            outf("@err usage: test sine <id 1..3> <amp counts> <hz 0.2..10> [cycles 1..50]");
            return;
        }
        if (b == 0 || b > servo_limit((int)a) || b < -servo_limit((int)a) || f_milli < 200 || f_milli > 10000 || cycles < 1 || cycles > 50) {
            outf("@err amplitude 1..%d counts, frequency 0.2..10 Hz, cycles 1..50", servo_limit((int)a));
            return;
        }
        tars_log_wave_t w;
        memset(&w, 0, sizeof w);
        w.kind = TLW_SINE;
        w.id = (uint8_t)a;
        w.amp = (int16_t)b;
        w.freq_mhz = (uint32_t)f_milli;
        w.cycles = (uint16_t)cycles;
        w.lead_ms = 400;
        w.tail_ms = 300;
        snprintf(args, sizeof args, "id=%d amp=%d freq_mhz=%d cycles=%d", (int)a, (int)b, (int)f_milli, (int)cycles);
        arm_wave(&w, "sine", args);
    } else if (!strcmp(what, "walk") || !strcmp(what, "turn")) {
        const bool walk = !strcmp(what, "walk");
        const int lim = walk ? WALK_MAX_VX : TURN_MAX_VYAW;
        int32_t delay = 0;
        if (argc < 4 || !tars_log_parse_i32(argv[2], &a) || !tars_log_parse_i32(argv[3], &b) || a == 0 || a > lim || a < -lim || b < 1 || b > 60 ||
            (argc > 4 && (!tars_log_parse_i32(argv[4], &delay) || delay < 0 || delay > 120)) || argc > 5) {
            outf("@err usage: test %s <%s counts -%d..%d, not 0> <seconds 1..60> [delay seconds 0..120]   (delay: the recording starts now, the robot moves later - unplug the USB cable meanwhile)",
                 what, walk ? "vx" : "vyaw", lim, lim);
            return;
        }
        if (!servo_use_ok()) return;
        if (S.own != OWN_NONE) {
            outf("@err servos are held (%s) - send 'release' first", own_name());
            return;
        }
        if (delay > 0) snprintf(args, sizeof args, "%s=%d sec=%d delay=%d", walk ? "vx" : "vyaw", (int)a, (int)b, (int)delay);
        else snprintf(args, sizeof args, "%s=%d sec=%d", walk ? "vx" : "vyaw", (int)a, (int)b);
        if (begin_record(what, args, b + 1 + delay, walk ? ACT_WALK : ACT_TURN, false, 0)) {
            if (delay > 0) {
                S.walk_start_us = tlp_now_us() + (int64_t)delay * 1000000;
                S.walk_val = (int)a;
                S.walk_ms = (int)b * 1000;
                S.walk_turn = walk ? 0 : 1;
                outf("@ok %s will start in %d s and run %d s (recording since now; the data stays in the robot's RAM: you may unplug the USB cable and plug it back after the walk, then type 'dump')",
                     what, (int)delay, (int)b);
            } else {
                if (walk) tlp_walk((int)a, (int)b * 1000);
                else tlp_turn((int)a, (int)b * 1000);
                outf("@ok %s started (%d s) - the robot WALKS, keep it on a safe surface; 'stop' ends it", what, (int)b);
            }
        }
    } else {
        outf("@err unknown test '%s'", what);
    }
}

static void cmd_pose(int argc, char **argv)
{
    int32_t c[3] = {0, 0, 0}, sec = POSE_DEFAULT_S;
    if (argc < 4) {
        outf("@err usage: pose <c1> <c2> <c3> [seconds, default %d, max %d]", POSE_DEFAULT_S, POSE_MAX_S);
        return;
    }
    for (int i = 0; i < 3; i++) {
        if (!tars_log_parse_i32(argv[1 + i], &c[i])) {
            outf("@err pose: '%s' is not a number", argv[1 + i]);
            return;
        }
        if (c[i] > servo_limit(i + 1) || c[i] < -servo_limit(i + 1)) {
            outf("@err pose: servo %d limited to +-%d counts", i + 1, servo_limit(i + 1));
            return;
        }
    }
    if (argc > 4 && (!tars_log_parse_i32(argv[4], &sec) || sec < 1 || sec > POSE_MAX_S)) {
        outf("@err pose: seconds 1..%d", POSE_MAX_S);
        return;
    }
    if (!servo_use_ok()) return;
    if (S.own == OWN_WAVE) {
        outf("@err a servo test is running - send 'stop'");
        return;
    }
    const float pos[3] = {(float)c[0], (float)c[1], (float)c[2]};
    tars_log_shell_hold_pose(pos, (uint32_t)sec * 1000u);
    outf("@ok pose %d %d %d held for %d s ('release' to end)", (int)c[0], (int)c[1], (int)c[2], (int)sec);
}

static void cmd_release(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    release_servos();
    outf("@ok release");
}

static void cmd_torque(int argc, char **argv)
{
    int32_t on = 1;
    if (argc < 2 || !tars_log_parse_i32(argv[1], &on) || (on != 0 && on != 1)) {
        outf("@err usage: torque 0|1");
        return;
    }
    if (on) {
        release_servos();
        outf("@ok torque on (legs snap back to the zero pose)");
        return;
    }
    if (!servo_use_ok()) return;
    if (S.own == OWN_WAVE) {
        outf("@err a servo test is running - send 'stop'");
        return;
    }
    const int64_t now = tlp_now_us();
    tlp_lock();
    S.own = OWN_LIMP;
    S.limp_deadline_us = now + (int64_t)LIMP_MAX_S * 1000000;
    S.limp_sent_us = 0;
    S.limp_next = 0;
    S.sched_reset = true;
    tlp_unlock();
    outf("@ok torque OFF - servos are limp for up to %d s ('torque 1' or 'release' ends it); try 'watch'", LIMP_MAX_S);
}

static void cmd_cal(int argc, char **argv)
{
    int32_t step = 0;
    if (!idle_or_busy("cal")) return;
    if (argc < 2 || !tars_log_parse_i32(argv[1], &step) || (step != 0 && step != 1)) {
        outf("@err usage: cal 0|1   (0 = upright & still, 1 = nose-down ~30 deg & still)");
        return;
    }
    if (!tlp_imu_ok()) {
        outf("@err IMU not initialised");
        return;
    }
    outf("# keep the robot still for 1 s ...");
    const bool ok = tlp_rl_calibrate((int)step);
    tlp_flush_msgs();
    outf(ok ? "@ok cal step=%d calibrated=%d" : "@err cal step=%d failed (calibrated=%d)", (int)step, tlp_rl_is_calibrated() ? 1 : 0);
}

static void cmd_learn(int argc, char **argv)
{
    int32_t sec = 30;
    if (argc < 2) {
        outf("@err usage: learn dry | learn on go [seconds 1..600, default 30] | learn off | learn auto off|on|walk go | learn log | learn tune ... | learn gait [0..6] [save] | learn straight [on|off] | learn motion [<name> [speed%%]|stop]");
        return;
    }
    if (!strcmp(argv[1], "motion")) {                            /* learn motion [<name> [speed 30..100] | stop]: a show motion where the robot stands (needs the learned mode ON) */
        char names[224] = "";
        for (int i = 0; i < tlp_rl_motion_count(); i++) {
            const size_t len = strlen(names);
            snprintf(names + len, sizeof names - len, "%s%s", i ? " " : "", tlp_rl_motion_name(i));
        }
        if (argc == 2) {
            outf("@ok learn motion: %s   (learn motion <name> [speed 30..100] | learn motion stop)", names);
            return;
        }
        if (argc == 3 && !strcmp(argv[2], "stop")) {
            tlp_rl_motion_stop();
            tlp_flush_msgs();
            outf("@ok motion stop requested (the hips go home first, then the lift)");
            return;
        }
        int32_t pct = 100;
        if (argc > 4 || (argc == 4 && (!tars_log_parse_i32(argv[3], &pct) || pct < 30 || pct > 100))) {
            outf("@err usage: learn motion <%s> [speed 30..100 %%] | learn motion stop", names);
            return;
        }
        const int r = tlp_rl_motion(argv[2], (int)pct);
        tlp_flush_msgs();
        if (r == 0) outf("@ok motion %s accepted, speed %d%% - the robot MOVES where it stands (flat, grippy surface, nothing around it); it starts when the robot stands still and upright; 'learn motion stop' or estop ends it", argv[2], (int)pct);
        else if (r == -2) outf("@err the learned mode is off: 'learn on go [sec]' first (it normally starts by itself ~4 s after power-up)");
        else if (r == -3) outf("@err busy: the robot is not standing (falling, getting up, or already performing a motion)");
        else outf("@err unknown motion '%s' (or speed outside 30..100): %s", argv[2], names);
        return;
    }
    if (!strcmp(argv[1], "log")) {                               /* what the learned mode did, also while no PC was connected */
        char line[110];
        int n = 0;
        for (int i = 0; tlp_rl_history(i, line, sizeof line); i++, n++) outf("# hist %s", line);
        outf("@ok learn log %d entries (RAM only: lost at power-off)", n);
        return;
    }
    if (!strcmp(argv[1], "gait")) {                              /* which walking gait set: 0 = v4 (original simulator), 1 = v5 (real-calibrated simulator), ... */
        if (argc > 2 && !strcmp(argv[2], "custom")) {            /* learn gait custom <f|b|t|r> <15 integers, millionths>: replace ONE gait of the current set (volatile) */
            int32_t m[15];
            bool ok = argc == 19 && argv[3][0] && !argv[3][1] && strchr("fbtr", argv[3][0]);
            for (int i = 0; ok && i < 15; i++) ok = tars_log_parse_i32(argv[4 + i], &m[i]);
            if (!ok) {
                outf("@err usage: learn gait custom <f|b|t|r> q0 a1 b1 a2 b2 z0 c1 d1 c2 d2 f kp_h kd_h kp_z kd_z   (15 integers = value x 1e6: rad, m, Hz, rad/rad, ...; f forward, b backward, t turn left, r turn right)");
                return;
            }
            const int r = tlp_rl_gait_custom(argv[3][0], m);
            tlp_flush_msgs();
            if (r == -2) {
                outf("@err a walking state is running: stop the walk first");
                return;
            }
            if (r != 0) {
                outf("@err a parameter is outside the range the gait search used (hip +-0.3 rad, lift +-12 mm, f 0.6..2.0 Hz, ...): not applied");
                return;
            }
        } else if (argc > 2) {
            int32_t idx = 0;
            if (!tars_log_parse_i32(argv[2], &idx) || idx < 0 || idx >= tlp_rl_gait_count() || (argc > 3 && strcmp(argv[3], "save")) || argc > 4) {
                outf("@err usage: learn gait [0..%d] [save]   (0 = v4, original simulator; 1 = v5, calibrated simulator; 2 = v5 + faster forward; 3 = v6 for 10 mm foot plates; 4 = v7, smoother forward; 5 = v8, v7 + smooth backward; 6 = v9, sinusoid hips)", tlp_rl_gait_count() - 1);
                return;
            }
            const int r = tlp_rl_gait_set((int)idx, argc > 3);
            tlp_flush_msgs();
            if (r == -2) {
                outf("@err a walking state is running: stop the walk first");
                return;
            }
        }
        {
            const unsigned cm = tlp_rl_gait_custom_mask();
            char cs[24] = "";
            if (cm) snprintf(cs, sizeof cs, " custom=%s%s%s%s", cm & 1 ? "f" : "", cm & 2 ? "b" : "", cm & 4 ? "t" : "", cm & 8 ? "r" : "");
            outf("@ok learn gait=%d (%s)%s", tlp_rl_gait_get(), tlp_rl_gait_name(tlp_rl_gait_get()), cs);
        }
        return;
    }
    if (!strcmp(argv[1], "getup")) {                             /* learn getup [kf|alt|nn] [save]: which get-up the robot uses after a fall */
        if (argc > 2) {
            int mode = -1;
            if (!strcmp(argv[2], "kf")) mode = 0;
            else if (!strcmp(argv[2], "alt")) mode = 1;
            else if (!strcmp(argv[2], "nn")) mode = 2;
            if (mode < 0 || argc > 4 || (argc == 4 && strcmp(argv[3], "save"))) {
                outf("@err usage: learn getup [kf|alt|nn] [save]   (kf: keyframe get-up; alt: learned network on the 1st and 3rd attempt, keyframes on the 2nd; nn: network only; 'save' keeps it over power cycles)");
                return;
            }
            const int r = tlp_rl_getup_mode(mode, argc == 4);
            tlp_flush_msgs();
            if (r == -3) {
                outf("@err this firmware has no get-up network compiled in");
                return;
            }
        }
        const int m = tlp_rl_getup_mode_get();
        outf("@ok learn getup=%s network=%s", m == 0 ? "kf" : (m == 1 ? "alt" : "nn"), tlp_rl_getup_nn_available() ? "compiled in" : "none");
        return;
    }
    if (!strcmp(argv[1], "speed")) {                             /* learn speed [on|off|max <70..100>]: the walking command's magnitude scales the gait amplitude (volatile) */
        if (argc > 2) {
            int32_t pct = 0;
            const bool is_max = !strcmp(argv[2], "max");
            if (is_max ? (argc != 4 || !tars_log_parse_i32(argv[3], &pct) || pct < 70 || pct > 100) : (argc > 3 || (strcmp(argv[2], "on") && strcmp(argv[2], "off")))) {
                outf("@err usage: learn speed [on|off|max <70..100>]   (on: the walk / turn command's magnitude 14..20 counts scales the gait amplitude 70..max %%, default max 90; off: always the full gait)");
                return;
            }
            if (is_max) tlp_rl_speed_max((int)pct);
            else tlp_rl_speed_map(!strcmp(argv[2], "on"));
            tlp_flush_msgs();
        }
        outf("@ok learn speed=%s max=%d", tlp_rl_speed_map_get() ? "on" : "off", tlp_rl_speed_max_get());
        return;
    }
    if (!strcmp(argv[1], "straight")) {                          /* learn straight [on|off|<kp%> <kd%> <trim per mille>]: heading hold of the forward / backward walks (volatile) */
        int cur[4];
        tlp_rl_straight_get(cur);
        if (argc > 2) {
            int32_t kp = cur[1], kd = cur[2], trim = cur[3];
            int on = cur[0];
            bool ok = true;
            if (argc == 3 && (!strcmp(argv[2], "on") || !strcmp(argv[2], "off"))) {
                on = !strcmp(argv[2], "on");
            } else if (argc >= 4 && argc <= 5) {                 /* explicit numbers switch it on */
                on = 1;
                ok = tars_log_parse_i32(argv[2], &kp) && tars_log_parse_i32(argv[3], &kd) && (argc < 5 || tars_log_parse_i32(argv[4], &trim));
                ok = ok && kp >= 0 && kp <= 300 && kd >= 0 && kd <= 100 && trim >= -300 && trim <= 300;
            } else {
                ok = false;
            }
            if (!ok) {
                outf("@err usage: learn straight [on|off | kp%% kd%% [trim_permille]]   (heading hold of the forward / backward walks from the gyro: stroke fraction x100 per rad of heading error 0..300 (default 150) and per rad/s 0..100 (default 0), trim -300..300 per mille, + = turn right; numbers switch it on; volatile)");
                return;
            }
            tlp_rl_straight(on, (int)kp, (int)kd, (int)trim);
            tlp_flush_msgs();
        }
        tlp_rl_straight_get(cur);
        outf("@ok learn straight=%s kp=%d%% kd=%d%% trim=%d permille", cur[0] ? "on" : "off", cur[1], cur[2], cur[3]);
        return;
    }
    if (!strcmp(argv[1], "tune")) {                              /* walking feedback tuning, volatile (experiments on the real robot) */
        int32_t v[5];
        int cur[5];
        tlp_rl_tune_get(cur);
        for (int i = 0; i < 5; i++) v[i] = cur[i];
        if (argc > 2) {
            if (argc > 7) {
                outf("@err usage: learn tune [kp%% 0..300] [kd%% 0..300] [lpf_hz 0..50] [amp%% 20..150] [rate_hz 50|100|250|500]   (no arguments: show; default 100 100 0 100 500; 'learn tune 100 100 0 100 50' = the searched gait at the simulated 50 Hz)");
                return;
            }
            static const int lo[5] = {0, 0, 0, 20, 50}, hi[5] = {300, 300, 50, 150, 500};
            for (int i = 0; i < 5 && i + 2 < argc; i++) {
                if (!tars_log_parse_i32(argv[i + 2], &v[i]) || v[i] < lo[i] || v[i] > hi[i] || (i == 4 && v[i] != 50 && v[i] != 100 && v[i] != 250 && v[i] != 500)) {
                    if (i == 4) outf("@err learn tune: argument 5 (controller rate) must be 50, 100, 250 or 500");
                    else outf("@err learn tune: argument %d must be %d..%d", i + 1, lo[i], hi[i]);
                    return;
                }
            }
            tlp_rl_tune((int)v[0], (int)v[1], (int)v[2], (int)v[3], (int)v[4]);
            tlp_flush_msgs();
        }
        tlp_rl_tune_get(cur);
        outf("@ok learn tune kp=%d%% kd=%d%% lpf=%dHz amp=%d%% hz=%d", cur[0], cur[1], cur[2], cur[3], cur[4]);
        return;
    }
    if (!strcmp(argv[1], "auto")) {                              /* persistent: the learned mode starts by itself at the next power-up */
        int mode = -1;
        if (argc >= 3 && !strcmp(argv[2], "off")) mode = 0;
        else if (argc >= 3 && !strcmp(argv[2], "on")) mode = 1;
        else if (argc >= 3 && !strcmp(argv[2], "walk")) mode = 2;
        if (mode < 0) {
            outf("@err usage: learn auto off | learn auto on | learn auto walk go   (now: auto=%d)", tlp_rl_autostart_get());
            return;
        }
        if (mode > 0 && !tlp_rl_is_calibrated()) {
            outf("@err IMU axes are not calibrated: run 'cal 0' then 'cal 1' first");
            return;
        }
        if (mode == 2 && (argc < 4 || strcmp(argv[3], "go"))) {
            outf("@err 'learn auto walk' lets BLE / voice commands make the robot walk with the learned gait: type 'learn auto walk go' to confirm");
            return;
        }
        tlp_rl_autostart(mode);
        tlp_flush_msgs();
        outf("@ok learn auto=%d (takes effect at the next power-up)", mode);
        return;
    }
    if (!strcmp(argv[1], "off")) {
        S.learn_deadline_us = 0;
        tlp_rl_mode(false, true);
    } else if (!strcmp(argv[1], "dry") || !strcmp(argv[1], "on")) {
        const bool dry = !strcmp(argv[1], "dry");
        if (!tlp_rl_is_calibrated()) {
            outf("@err IMU axes are not calibrated: run 'cal 0' then 'cal 1' first");
            return;
        }
        if (!dry && (argc < 3 || strcmp(argv[2], "go"))) {
            outf("@err 'learn on' makes the robot move by itself: type 'learn on go [seconds]' to confirm (hold it / use a soft surface; it switches itself off after the time)");
            return;
        }
        if (!dry && argc > 3 && (!tars_log_parse_i32(argv[3], &sec) || sec < 1 || sec > 600)) {
            outf("@err usage: learn on go [seconds 1..600]");
            return;
        }
        if (!servo_use_ok()) return;
        tlp_rl_mode(true, dry);
        S.learn_deadline_us = dry ? 0 : tlp_now_us() + (int64_t)sec * 1000000;
    } else {
        outf("@err usage: learn dry | learn on go [seconds] | learn off | learn auto off|on|walk go | learn log | learn tune ... | learn gait [0..6] [save] | learn straight [on|off] | learn motion [<name> [speed%%]|stop]");
        return;
    }
    tlp_flush_msgs();
    char rl[260];
    rl[0] = 0;
    tlp_rl_status(rl, sizeof rl);
    outf("@ok learn %s", rl);
}

static void cmd_map(int argc, char **argv)
{
    int32_t a = 0, b = 0, c = 0, d = 0;
    if (argc == 3 && tars_log_parse_i32(argv[1], &a) && tars_log_parse_i32(argv[2], &b) && (a == 0 || a == 1) && (b == 0 || b == 1)) {
        tlp_rl_map(a != 0, b != 0);                        /* map <flip 0|1> <swap_lr 0|1> */
    } else if (argc == 5 && tars_log_parse_i32(argv[1], &a) && tars_log_parse_i32(argv[2], &b) && tars_log_parse_i32(argv[3], &c) &&
               tars_log_parse_i32(argv[4], &d) && (a == 1 || a == -1) && (b == 1 || b == -1) && (c == 1 || c == -1) && (d == 0 || d == 1)) {
        tlp_rl_map3((int)a, (int)b, (int)c, d != 0);      /* map <sign ID1> <sign ID2> <sign ID3> <swap_lr 0|1>, +1 = hypothesis (a) */
    } else {
        outf("@err usage: map <flip 0|1> <swap_lr 0|1>   or   map <+1|-1 for ID1> <+1|-1 for ID2> <+1|-1 for ID3 (slider)> <swap_lr 0|1>");
        return;
    }
    tlp_flush_msgs();
    char rl[260];
    rl[0] = 0;
    tlp_rl_status(rl, sizeof rl);
    outf("@ok map %s", rl);
}

/* 'walk <vx> <sec>' / 'turn <vyaw> <sec>': the gait runs for <sec> seconds WITHOUT a recording session; use it together with `watch`, which streams the
 * supply voltage / servo load / attitude to the PC in real time (a recording lives in RAM and is lost when the robot loses power). */
static void cmd_walk_or_turn(int argc, char **argv, bool walk)
{
    int32_t a = 0, b = 0;
    const int lim = walk ? WALK_MAX_VX : TURN_MAX_VYAW;
    if (argc < 3 || !tars_log_parse_i32(argv[1], &a) || !tars_log_parse_i32(argv[2], &b) || a == 0 || a > lim || a < -lim || b < 1 || b > 60) {
        outf("@err usage: %s <%s counts -%d..%d, not 0> <seconds 1..60>   (no recording; use with 'watch')", walk ? "walk" : "turn", walk ? "vx" : "vyaw", lim, lim);
        return;
    }
    if (!servo_use_ok()) return;
    if (S.own != OWN_NONE) {
        outf("@err servos are held (%s) - send 'release' first", own_name());
        return;
    }
    if (walk) tlp_walk((int)a, (int)b * 1000);
    else tlp_turn((int)a, (int)b * 1000);
    S.plain_move_active = 1;
    outf("@ok %s started (%d s) - the robot MOVES, keep it on a safe surface; 'estop' ends it", walk ? "walk" : "turn", (int)b);
}

static void cmd_walk(int argc, char **argv) { cmd_walk_or_turn(argc, argv, true); }
static void cmd_turn(int argc, char **argv) { cmd_walk_or_turn(argc, argv, false); }

static void cmd_selftest(int argc, char **argv)
{
    int32_t step = 0;
    if (argc < 2 || !tars_log_parse_i32(argv[1], &step) || step < 0 || step > 3) {
        outf("@err usage: selftest 0..3   (1 = feet forward, 2 = legs up, 3 = zero pose, 0 = release)");
        return;
    }
    if (!servo_use_ok()) return;
    if (S.own == OWN_WAVE) {
        outf("@err a servo test is running - send 'stop'");
        return;
    }
    tlp_rl_selftest((int)step);
    tlp_flush_msgs();
    outf("@ok selftest %d", (int)step);
}

/* ---- IMU (QMI8658) register diagnostics ------------------------------------------------------------------------ */
static void imu_hex_line(const char *label, uint8_t reg0, int n)
{
    uint8_t b[16];
    char line[120];
    size_t k = 0;
    if (n > 16) n = 16;
    if (!tlp_imu_read(reg0, b, (size_t)n)) {
        outf("# imu 0x%02x: I2C read failed", reg0);
        return;
    }
    k += (size_t)snprintf(line + k, sizeof line - k, "# imu 0x%02x..0x%02x", reg0, reg0 + n - 1);
    for (int i = 0; i < n && k < sizeof line - 4; i++) k += (size_t)snprintf(line + k, sizeof line - k, " %02x", b[i]);
    outf("%s   %s", line, label);
}

static void imu_sample_line(void)
{
    uint8_t b[14];
    if (!tlp_imu_read(0x33, b, sizeof b)) {
        outf("# imu: I2C read failed");
        return;
    }
    const int16_t t = (int16_t)(b[0] | (b[1] << 8)), ax = (int16_t)(b[2] | (b[3] << 8)), ay = (int16_t)(b[4] | (b[5] << 8)),
                  az = (int16_t)(b[6] | (b[7] << 8)), gx = (int16_t)(b[8] | (b[9] << 8)), gy = (int16_t)(b[10] | (b[11] << 8)),
                  gz = (int16_t)(b[12] | (b[13] << 8));
    sh_line_t l;
    ln_reset(&l);
    ln_s(&l, "# imu raw acc=");
    ln_i(&l, ax); ln_s(&l, ","); ln_i(&l, ay); ln_s(&l, ","); ln_i(&l, az);
    ln_s(&l, " gyro=");
    ln_i(&l, gx); ln_s(&l, ","); ln_i(&l, gy); ln_s(&l, ","); ln_i(&l, gz);
    ln_s(&l, " temp=");
    ln_m(&l, div_round((int32_t)t * 1000, 256), 1);
    ln_s(&l, "C");
    ln_send(&l);
}

static void imu_regs(void)
{
    imu_hex_line("WHO_AM_I REV CTRL1 CTRL2(acc fs/odr) CTRL3(gyr fs/odr) CTRL4 CTRL5 CTRL6 CTRL7 CTRL8 CTRL9", 0x00, 11);
    imu_hex_line("CAL1..CAL4 (L,H)", 0x0B, 8);
    imu_hex_line("STATUSINT STATUS0 STATUS1 TIMESTAMP(3) TEMP(2)", 0x2D, 8);
    imu_hex_line("AX AY AZ GX GY GZ (int16 little endian)", 0x35, 12);
    imu_sample_line();
}

static void cmd_imu(int argc, char **argv)
{
    const char *sub = argc > 1 ? argv[1] : "regs";
    uint8_t who = 0;
    if (!tlp_imu_read(0x00, &who, 1)) {
        outf("@err imu: no I2C access to the QMI8658 (bus not initialised?)");
        return;
    }
    if (!strcmp(sub, "regs")) {
        imu_regs();
    } else if (!strcmp(sub, "raw")) {
        int32_t n = 20;
        if (argc > 2 && (!tars_log_parse_i32(argv[2], &n) || n < 1 || n > 100)) {
            outf("@err usage: imu raw [samples 1..100]");
            return;
        }
        for (int i = 0; i < n; i++) {
            imu_sample_line();
            tlp_delay_ms(20);
        }
    } else if (!strcmp(sub, "reinit")) {
        /* soft reset of the sensor, then the same configuration the stock driver writes (imu.cc) */
        tlp_imu_write(0x60, 0xB0);
        tlp_delay_ms(30);
        uint8_t rst = 0;
        tlp_imu_read(0x4D, &rst, 1);
        outf("# imu soft reset: RST_RESULT=0x%02x (0x80 = done)", rst);
        tlp_imu_write(0x02, 0x40);
        tlp_imu_write(0x08, 0x03);
        tlp_imu_write(0x03, 0x04);
        tlp_imu_write(0x04, 0x64);
        tlp_imu_write(0x06, 0x00);
        tlp_delay_ms(80);
        imu_regs();
    } else if (!strcmp(sub, "fs")) {
        int32_t fs = 0;
        if (argc < 3 || !tars_log_parse_i32(argv[2], &fs) || fs < 0 || fs > 3) {
            outf("@err usage: imu fs <0..3>   accelerometer full scale +-2/4/8/16 g (the firmware assumes 0; 'imu reinit' or a reboot restores it)");
            return;
        }
        tlp_imu_write(0x03, (uint8_t)((fs << 4) | 0x04));
        tlp_delay_ms(60);
        outf("# imu accelerometer full scale index %d (counts per g: %d); the stock scaling still assumes 16384", (int)fs, 16384 >> fs);
        imu_regs();
    } else {
        outf("@err usage: imu [regs | raw [n] | reinit | fs <0..3>]");
        return;
    }
    outf("@ok imu %s", sub);
}

static void cmd_link(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "usb") && strcmp(argv[1], "console"))) {
        outf("@err usage: link usb|console   (console = slower fallback through the normal console output)");
        return;
    }
    tlp_set_link(!strcmp(argv[1], "console"));
    outf("@ok link %s", argv[1]);
}

static void cmd_baud(int argc, char **argv)
{
    static const int32_t kRates[] = {115200, 230400, 460800, 921600, 1500000, 2000000};
    int32_t rate = 0;
    if (argc >= 2 && !strcmp(argv[1], "ok")) {            /* the PC switched too: keep the new rate */
        tlp_baud_confirm();
        outf("@ok baud confirmed");
        return;
    }
    bool valid = false;
    if (argc >= 2 && tars_log_parse_i32(argv[1], &rate)) {
        for (size_t i = 0; i < sizeof kRates / sizeof kRates[0]; i++) valid = valid || rate == kRates[i];
    }
    if (!valid) {
        outf("@err usage: baud 115200|230400|460800|921600|1500000|2000000   (then switch the PC side and send 'baud ok' within 8 s)");
        return;
    }
    outf("@ok baud %d (switching now; send 'baud ok' at the new rate within 8 s or I go back to 115200)", (int)rate);
    if (!tlp_set_baud((uint32_t)rate)) outf("@err baud: the UART refused that rate");
}

static void cmd_reboot(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    outf("@ok reboot");
    tlp_delay_ms(150);
    tlp_reboot();
}

static void cmd_errors(int argc, char **argv)
{
    (void)argv;
    if (argc != 1) {
        outf("@err usage: errors");
        return;
    }
    tlp_error_log(outf);
}

static const struct {
    const char *name;
    void (*fn)(int, char **);
} kCmds[] = {
    {"help", cmd_help},       {"?", cmd_help},       {"hello", cmd_hello}, {"info", cmd_hello}, {"status", cmd_status},
    {"scan", cmd_scan},       {"watch", cmd_watch},  {"rec", cmd_rec},     {"live", cmd_live},  {"stop", cmd_stop}, {"estop", cmd_estop},
    {"dump", cmd_dump},       {"clear", cmd_clear},  {"mark", cmd_mark},   {"test", cmd_test},  {"pose", cmd_pose},
    {"release", cmd_release}, {"torque", cmd_torque}, {"cal", cmd_cal},    {"learn", cmd_learn}, {"map", cmd_map}, {"walk", cmd_walk}, {"turn", cmd_turn},
    {"selftest", cmd_selftest}, {"imu", cmd_imu}, {"link", cmd_link}, {"baud", cmd_baud}, {"reboot", cmd_reboot}, {"errors", cmd_errors},
};

static void exec_line(char *line)
{
    char *argv[20];
    const int argc = tars_log_tokenize(line, argv, 20);
    if (argc == 0) return;
    for (char *p = argv[0]; *p; p++) {
        if (*p >= 'A' && *p <= 'Z') *p = (char)(*p + 32);
    }
    for (size_t i = 0; i < sizeof kCmds / sizeof kCmds[0]; i++) {
        if (!strcmp(argv[0], kCmds[i].name)) {
            kCmds[i].fn(argc, argv);
            return;
        }
    }
    outf("@err unknown command '%.24s' (type help)", argv[0]);
}

void tars_log_shell_feed(const uint8_t *data, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const uint8_t c = data[i];
        if (c == 0x03) {                                   /* Ctrl-C = emergency stop */
            char estop[] = "estop";
            S.line_n = 0;
            S.line_over = false;
            exec_line(estop);
            continue;
        }
        if (c == '\r' || c == '\n') {
            if (S.line_over) {
                outf("@err line too long");
            } else if (S.line_n > 0) {
                S.line[S.line_n] = 0;
                exec_line(S.line);
            }
            S.line_n = 0;
            S.line_over = false;
            continue;
        }
        if (c == 0x08 || c == 0x7F) {
            if (S.line_n) S.line_n--;
            continue;
        }
        if (c < 32 || c >= 127) continue;                  /* line noise */
        if (S.line_n < sizeof S.line - 1) S.line[S.line_n++] = (char)c;
        else S.line_over = true;
    }
}

/* ---------------------------------------------------------------------------------------------------------------
 * periodic work (log task)
 * --------------------------------------------------------------------------------------------------------------- */
bool tars_log_shell_idle(void)
{
    return S.mode == MODE_IDLE;
}

bool tars_log_shell_poll(void)
{
    const int64_t now = tlp_now_us();
    bool more = false;

    if (S.dump_aborted_msg && S.mode == MODE_IDLE) {          /* repeated until the host is reading again and takes it */
        static const char kMsg[] = "# the last dump was cut short (host not reading); type 'dump' to send it again\n";
        if (tlp_out(kMsg, sizeof kMsg - 1)) S.dump_aborted_msg = false;
    }

    if (S.learn_deadline_us && now >= S.learn_deadline_us) {   /* 'learn on go <sec>' ran out: back to the zero pose, classic gait */
        S.learn_deadline_us = 0;
        tlp_rl_mode(false, true);
        tlp_flush_msgs();
        outf("# learned mode switched itself off (time is up); the robot returns to the zero pose");
    }

    if (S.walk_start_us && now >= S.walk_start_us) {              /* 'test walk|turn ... <delay>': the delay is over, the robot starts moving */
        S.walk_start_us = 0;
        if (S.mode == MODE_REC) {
            if (S.walk_turn) tlp_turn(S.walk_val, S.walk_ms);
            else tlp_walk(S.walk_val, S.walk_ms);
        }
    }

    if (S.mode == MODE_REC) {
        if (S.deadline_us && now >= S.deadline_us) {
            finish_recording("time up");
        } else if (S.is_wave) {
            tlp_lock();
            const bool done = S.wave_done && (now - S.wave_done_us) >= WAVE_SETTLE_US;
            const bool lost = S.own != OWN_WAVE && !S.wave_done;      /* released / aborted before the end */
            tlp_unlock();
            if (done) finish_recording("test done");
            else if (lost) finish_recording("test aborted");
        } else if (S.ring_ok && tars_log_ring_head(&S.ring) >= S.ring.cap) {
            finish_recording("buffer full");
        }
    }

    switch (S.mode) {
    case MODE_WATCH:
        if (now >= S.watch_next_us) {
            S.watch_next_us = now + (int64_t)S.watch_period_ms * 1000;
            print_watch(now);
        }
        break;
    case MODE_LIVE:
        more = live_step();
        break;
    case MODE_SCAN:
        if (now >= S.deadline_us) finish_scan();
        break;
    case MODE_DUMP:
        more = dump_step(now);
        break;
    default:
        break;
    }
    return more;
}

/* ---------------------------------------------------------------------------------------------------------------
 * robot task side
 * --------------------------------------------------------------------------------------------------------------- */
bool tars_log_shell_bus_active(void)
{
    return S.bus_fast || S.own != OWN_NONE;
}

bool tars_log_shell_owns_servos(void)
{
    return S.own != OWN_NONE;
}

void tars_log_shell_tick(int64_t now, tars_log_tick_out_t *out)
{
    memset(out, 0, sizeof *out);
    bool fast, reset;
    int focus;

    tlp_lock();
    switch (S.own) {
    case OWN_WAVE: {
        if (!S.wave_started) {
            S.wave_t0_us = now;
            S.wave_started = true;
        }
        const int32_t t_ms = (int32_t)((now - S.wave_t0_us) / 1000);
        out->write = 1;                                   /* positions are 0 once the test is over */
        out->cmd_src = 1;
        if (!tars_log_wave_eval(&S.wave, t_ms, out->pos) && !S.wave_done) {
            S.wave_done = true;
            S.wave_done_us = now;
        }
        break;
    }
    case OWN_POSE:
        if (now >= S.pose_deadline_us) {
            S.own = OWN_NONE;
        } else if (now - S.pose_sent_us >= POSE_RESEND_US) {
            out->write = 1;
            out->cmd_src = 1;
            for (int i = 0; i < 3; i++) out->pos[i] = S.pose[i];
            S.pose_sent_us = now;
        }
        break;
    case OWN_LIMP:
        if (now >= S.limp_deadline_us) {
            S.own = OWN_RESTORE;
            S.restore_n = 0;
        } else if (now - S.limp_sent_us >= LIMP_SEND_US) {
            out->enable_id = S.limp_next + 1;
            out->enable_val = 0;
            S.limp_next = (S.limp_next + 1) % 3;
            S.limp_sent_us = now;
        }
        break;
    case OWN_RESTORE:
        out->enable_id = (S.restore_n % 3) + 1;
        out->enable_val = 1;
        if (++S.restore_n >= 9) S.own = OWN_NONE;
        break;
    default:
        break;
    }
    fast = S.bus_fast;
    focus = S.focus;
    reset = S.sched_reset;
    S.sched_reset = false;
    tlp_unlock();

    /* read-back: at most one bus request per cycle; the bus voltage has priority when it is due */
    if (reset) {
        for (int i = 0; i < 3; i++) S.next_due[i] = now + (int64_t)i * 3000;
        S.volt_due_us = now + 100000;
    }
    if (now >= S.volt_due_us) {
        out->read_volt = 1;
        S.volt_due_us = now + (fast ? VOLT_FAST_US : VOLT_SLOW_US);
        return;
    }
    int best = -1;
    for (int i = 0; i < 3; i++) {
        if (now >= S.next_due[i] && (best < 0 || S.next_due[i] < S.next_due[best])) best = i;
    }
    if (best >= 0) {
        int64_t period = FB_SLOW_US;
        if (fast) period = focus ? ((best + 1 == focus) ? FB_FOCUS_US : FB_OTHER_US) : FB_PERIOD_US;
        out->read_id = best + 1;
        S.next_due[best] += period;
        if (S.next_due[best] < now) S.next_due[best] = now + period;      /* fell behind: do not burst */
        S.fb[best].req_us = now;
    }
}

void tars_log_shell_abort_servo_use(void)
{
    tlp_lock();
    S.own = OWN_NONE;
    tlp_unlock();
}

void tars_log_shell_hold_pose(const float pos[3], uint32_t timeout_ms)
{
    float p[3];
    for (int i = 0; i < 3; i++) {
        const float lim = (float)servo_limit(i + 1);
        p[i] = pos[i] > lim ? lim : (pos[i] < -lim ? -lim : pos[i]);
    }
    const int64_t now = tlp_now_us();
    tlp_lock();
    if (S.own != OWN_WAVE) {
        for (int i = 0; i < 3; i++) S.pose[i] = p[i];
        S.pose_deadline_us = now + (int64_t)timeout_ms * 1000;
        S.pose_sent_us = 0;
        S.own = OWN_POSE;
        S.sched_reset = true;
    }
    tlp_unlock();
}

void tars_log_shell_release(void)
{
    release_servos();
}

/* ---------------------------------------------------------------------------------------------------------------
 * data hooks (any task)
 * --------------------------------------------------------------------------------------------------------------- */
void tars_log_shell_on_imu(const tars_log_imu_t *s, int64_t now)
{
    S.imu = *s;
    S.imu_t_us = now;
    if (rec_get()) {
        int16_t v[12];
        for (int i = 0; i < 3; i++) {
            v[i] = s->acc[i];
            v[3 + i] = s->gyro[i];
            v[6 + i] = s->up[i];
        }
        v[9] = s->flags;
        v[10] = s->state;
        v[11] = (int16_t)(((int)s->vx << 8) | (int)(uint8_t)s->vyaw);
        tars_log_ring_push(&S.ring, now, TLE_IMU, 0, v, 12);
    }
}

void tars_log_shell_on_servo(int id, int16_t pos, int16_t spd, int16_t load, int64_t now)
{
    if (id < 1 || id > 3) return;
    sh_fb_t *f = &S.fb[id - 1];
    int32_t lat = 0;
    if (f->req_us && now - f->req_us < 100000) lat = (int32_t)(now - f->req_us);
    if (lat > 32767) lat = 32767;
    f->pos = pos;
    f->spd = spd;
    f->load = load;
    f->t_us = now;
    f->req_us = 0;
    if (rec_get()) {
        int16_t v[4] = {pos, spd, load, (int16_t)lat};
        tars_log_ring_push(&S.ring, now, TLE_SERVO, (uint8_t)id, v, 4);
    }
}

void tars_log_shell_on_cmd(const int16_t pos[3], int16_t speed, int src, int64_t now)
{
    for (int i = 0; i < 3; i++) S.cmd[i] = pos[i];
    S.cmd_speed = speed;
    S.cmd_t_us = now;
    if (rec_get() && tars_log_cmd_should_log(&S.cmd_filter, now, pos)) {
        int16_t v[4] = {pos[0], pos[1], pos[2], speed};
        tars_log_ring_push(&S.ring, now, TLE_CMD, (uint8_t)src, v, 4);
    }
}

void tars_log_shell_on_volt(int16_t centivolt, int64_t now)
{
    S.volt_cv = centivolt;
    S.volt_t_us = now;
    if (rec_get()) {
        int16_t v[1] = {centivolt};
        tars_log_ring_push(&S.ring, now, TLE_VOLT, 0, v, 1);
    }
}

void tars_log_shell_on_msg(const char *text, int64_t now)
{
    if (rec_get()) push_text(now, text);
}

void tars_log_shell_init(void)
{
    memset(&S, 0, sizeof S);
    S.mode = MODE_IDLE;
}
