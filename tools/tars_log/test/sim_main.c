/*
 * sim_main.c - host-side simulator of the Tars robot around the REAL log shell (main/boards/tars/tars_log_shell.c + _core.c).
 *
 *   tars_log_sim            talk on stdin/stdout (type commands, read the protocol)
 *   tars_log_sim --pty      create a pseudo terminal, print "PTY=/dev/ttysNNN" on stderr and talk through it (what pyserial opens)
 *
 * environment: SIM_SPEED (virtual time speed-up, default 1), SIM_RING (ring events, default 8192), SIM_DEAD (servo id that never answers),
 *              SIM_CALIB (1 = servo calibration mode active), SIM_OUT_FAIL_AFTER=N + SIM_OUT_STALL_MS=M (the host stops reading after N bytes for M ms)
 */
#define _DARWIN_C_SOURCE 1
#define _DEFAULT_SOURCE 1
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "tars_log_port.h"
#include "tars_log_shell.h"

/* ------------------------------------------------------------------------------------------------ time / io */
static double g_speed = 1.0;
static int g_out_fd = 1, g_in_fd = 0;
static long g_out_bytes = 0, g_out_fail_after = -1;
static int g_dead = 0, g_calib = 0;
static uint32_t g_ring_cap = 8192;
static struct timespec g_t0;

static int64_t real_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec - g_t0.tv_sec) * 1000000 + (ts.tv_nsec - g_t0.tv_nsec) / 1000;
}
static int64_t g_vnow = 0;     /* virtual time driven by the main loop */
int64_t tlp_now_us(void) { return g_vnow; }
void tlp_delay_ms(uint32_t ms) { (void)ms; }
void tlp_lock(void) {}
void tlp_unlock(void) {}
void tlp_flush_msgs(void) {}
static int g_link_console = 0;
void tlp_set_link(int console) { g_link_console = console; }
static uint32_t g_baud = 115200;
static int g_baud_confirmed = 1;
bool tlp_set_baud(uint32_t baud) { g_baud = baud; g_baud_confirmed = (baud == 115200); return true; }
void tlp_baud_confirm(void) { g_baud_confirmed = 1; }

static int64_t g_stall_until = 0;
static int g_stall_started = 0;
static long g_stall_ms = 1500;

static void start_stall(void)
{
    g_stall_started = 1;
    g_stall_until = g_vnow + (int64_t)g_stall_ms * 1000;
}

/* bytes accepted; SIM_OUT_FAIL_AFTER=N: after N bytes the "host" stops reading for SIM_OUT_STALL_MS (virtual) */
size_t tlp_write(const char *s, size_t n)
{
    if (g_vnow < g_stall_until) return 0;
    size_t accept = n;
    if (g_out_fail_after >= 0 && !g_stall_started) {
        if (g_out_bytes >= g_out_fail_after) {
            start_stall();
            return 0;
        }
        if (g_out_bytes + (long)n > g_out_fail_after) accept = (size_t)(g_out_fail_after - g_out_bytes);
    }
    size_t off = 0;
    while (off < accept) {
        ssize_t w = write(g_out_fd, s + off, accept - off);
        if (w < 0) {
            struct pollfd p = {g_out_fd, POLLOUT, 0};
            poll(&p, 1, 5);
            continue;
        }
        off += (size_t)w;
    }
    g_out_bytes += (long)accept;
    if (g_out_fail_after >= 0 && !g_stall_started && g_out_bytes >= g_out_fail_after) start_stall();
    return accept;
}

bool tlp_out(const char *s, size_t n) { return tlp_write(s, n) == n; }

tars_log_ev_t *tlp_alloc_events(uint32_t *cap)
{
    *cap = g_ring_cap;
    return (tars_log_ev_t *)calloc(g_ring_cap, sizeof(tars_log_ev_t));
}

/* ------------------------------------------------------------------------------------------------ robot model */
#define ZERO0 430
#define ZERO1 540
#define ZERO2 492
static const int16_t kZero[3] = {ZERO0, ZERO1, ZERO2};
static const int kMask[3] = {1, 1, -1};

typedef struct {
    double pos, vel, des;
    double q_t[64], q_d[64];       /* command delay line */
    int q_n;
    bool limp;
} sim_servo_t;
static sim_servo_t sv[3];
static bool g_imu_on = false;
static bool g_rl_on;
static double g_vx = 0, g_vyaw = 0;
static int64_t g_gait_until = 0, g_gait_t0 = 0;
static float g_cmd_pos[3] = {0, 0, 0};

typedef struct { int64_t due; int kind; int id; } pending_t;   /* kind 0 = state reply, 1 = voltage reply */
static pending_t pend[64];
static int npend = 0;

int tlp_servo_mask(int i) { return kMask[i]; }
int16_t tlp_zero_pos(int i) { return kZero[i]; }
int tlp_calibrating(void) { return g_calib; }
bool tlp_imu_ok(void) { return true; }
void tlp_imu_stream(bool on) { g_imu_on = on; }

/* fake QMI8658 register file */
static uint8_t g_qmi[256];
static void qmi_defaults(void)
{
    memset(g_qmi, 0, sizeof g_qmi);
    g_qmi[0x00] = 0x05; g_qmi[0x01] = 0x7c;
    g_qmi[0x02] = 0x40; g_qmi[0x03] = 0x04; g_qmi[0x04] = 0x64; g_qmi[0x06] = 0x00; g_qmi[0x08] = 0x03;
    g_qmi[0x4D] = 0x80;
}
static void qmi_data(void)
{
    /* temp 25.5 C, acc (1140, 16335, 1190) [z = -32768 with SIM_STUCK_Z], gyro (7, 171, -8) */
    const int16_t v[7] = {(int16_t)(25.5 * 256), 1140, 16335, getenv("SIM_STUCK_Z") ? (int16_t)-32768 : (int16_t)1190, 7, 171, -8};
    for (int i = 0; i < 7; i++) {
        g_qmi[0x33 + 2 * i] = (uint8_t)(v[i] & 0xFF);
        g_qmi[0x34 + 2 * i] = (uint8_t)((v[i] >> 8) & 0xFF);
    }
}
bool tlp_imu_read(uint8_t reg, uint8_t *buf, size_t n)
{
    if (g_qmi[0x00] == 0) qmi_defaults();
    qmi_data();
    for (size_t i = 0; i < n; i++) buf[i] = g_qmi[(reg + i) & 0xFF];
    return true;
}
bool tlp_imu_write(uint8_t reg, uint8_t val)
{
    if (g_qmi[0x00] == 0) qmi_defaults();
    if (reg == 0x60 && val == 0xB0) {
        qmi_defaults();
        return true;
    }
    g_qmi[reg] = val;
    return true;
}

static void write_pos(const float off[3], int src)
{
    int16_t reg[3];
    for (int i = 0; i < 3; i++) {
        double d = kMask[i] * off[i] + kZero[i];
        if (d < 1) d = 1;
        if (d > 1023) d = 1023;
        reg[i] = (int16_t)d;
        if (sv[i].q_n < 64) {
            sv[i].q_t[sv[i].q_n] = (double)g_vnow + 6000.0;     /* 6 ms transport + servo delay */
            sv[i].q_d[sv[i].q_n] = d;
            sv[i].q_n++;
        }
        g_cmd_pos[i] = off[i];
    }
    tars_log_shell_on_cmd(reg, 0, src, g_vnow);
}

static void servo_step(double dt)
{
    for (int i = 0; i < 3; i++) {
        sim_servo_t *s = &sv[i];
        int k = 0;
        while (k < s->q_n && s->q_t[k] <= (double)g_vnow) {
            s->des = s->q_d[k];
            k++;
        }
        if (k) {
            memmove(s->q_t, s->q_t + k, (size_t)(s->q_n - k) * sizeof(double));
            memmove(s->q_d, s->q_d + k, (size_t)(s->q_n - k) * sizeof(double));
            s->q_n -= k;
        }
        if (s->limp) {
            /* somebody turns the leg by hand: slow sine */
            double target = kZero[i] + 60.0 * sin(2 * M_PI * 0.3 * (double)g_vnow * 1e-6 + i);
            s->vel = (target - s->pos) / dt * 0.0;
            s->pos = target;
            continue;
        }
        const double err = s->des - s->pos;
        double vcmd = err * 40.0;
        if (vcmd > 2048) vcmd = 2048;
        if (vcmd < -2048) vcmd = -2048;
        const double amax = 30000.0 * dt;
        double dv = vcmd - s->vel;
        if (dv > amax) dv = amax;
        if (dv < -amax) dv = -amax;
        s->vel += dv;
        s->pos += s->vel * dt;
    }
}

static void servo_reply(int id)
{
    sim_servo_t *s = &sv[id - 1];
    int p = (int)lround(s->pos);
    double err = s->des - s->pos;
    int load = (int)lround(err * 8.0);
    if (s->limp) load = 0;
    if (load > 1000) load = 1000;
    if (load < -1000) load = -1000;
    int mag = (int)fabs(s->vel);
    if (mag > 32767) mag = 32767;
    uint16_t spd_raw = (uint16_t)((s->vel < 0 ? 0x8000 : 0) | mag);
    uint16_t load_raw = (uint16_t)((load < 0 ? 0x400 : 0) | (abs(load) & 0x3FF));
    tars_log_shell_on_servo(id, (int16_t)p, (int16_t)spd_raw, (int16_t)load_raw, g_vnow);
}

static uint32_t rng = 12345;
static double noise(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xFFFF) / 65536.0 - 0.5;
}

static double g_up_x = 0.03, g_up_z = 0.999;
static int64_t g_ctl_next = 0;
static void sim_read_ctl(void)
{
    const char *path = getenv("SIM_CTL_FILE");
    if (!path || g_vnow < g_ctl_next) return;
    g_ctl_next = g_vnow + 50000;
    FILE *f = fopen(path, "r");
    if (!f) return;
    double x, z;
    if (fscanf(f, "%lf %lf", &x, &z) == 2) {
        g_up_x = x;
        g_up_z = z;
    }
    fclose(f);
}

static void imu_sample(void)
{
    sim_read_ctl();
    const double t = (double)g_vnow * 1e-6;
    const double walk = (fabs(g_vx) + fabs(g_vyaw)) > 0.5 ? 1.0 : 0.0;
    double ax = 0.30 + 0.9 * walk * sin(2 * M_PI * 1.25 * t) + 0.02 * noise();
    double ay = -0.10 + 0.4 * walk * cos(2 * M_PI * 1.25 * t) + 0.02 * noise();
    double az = 9.78 + 0.05 * noise();
    tars_log_imu_t s;
    s.acc[0] = (int16_t)lround(ax / 9.8 * 16384);
    s.acc[1] = (int16_t)lround(ay / 9.8 * 16384);
    s.acc[2] = getenv("SIM_STUCK_Z") ? (int16_t)-32768 : (int16_t)lround(az / 9.8 * 16384);
    s.gyro[0] = (int16_t)lround((0.4 + 25.0 * walk * sin(2 * M_PI * 1.25 * t) + 0.3 * noise()) * 32);
    s.gyro[1] = (int16_t)lround((-0.2 + 8.0 * walk * cos(2 * M_PI * 1.25 * t) + 0.3 * noise()) * 32);
    s.gyro[2] = (int16_t)lround((0.1 + 0.3 * noise()) * 32);
    s.up[0] = (int16_t)lround(g_up_x * 10000);
    s.up[1] = (int16_t)lround(-0.01 * 10000);
    s.up[2] = (int16_t)lround(g_up_z * 10000);
    s.flags = (uint8_t)(1 | (getenv("SIM_STUCK_Z") ? 32 : 0));
    s.state = g_rl_on ? (int8_t)(g_up_z < 0.5 ? 5 : 0) : (int8_t)-1;       /* learned supervisor: FALLEN when lying, else STAND */
    s.vx = (int8_t)g_vx;
    s.vyaw = (int8_t)g_vyaw;
    tars_log_shell_on_imu(&s, g_vnow);
}

/* classic gait as in robot_action.cc */
static void classic_gait(void)
{
    if (g_gait_until && g_vnow >= g_gait_until) {
        g_vx = g_vyaw = 0;
        g_gait_until = 0;
        g_gait_t0 = 0;
    }
    float pos[3] = {0, 0, 0};
    if (fabs(g_vx) >= 0.5 || fabs(g_vyaw) >= 0.5) {
        if (!g_gait_t0) g_gait_t0 = g_vnow;
        const double ph = 1.25 * ((double)(g_vnow - g_gait_t0) * 1e-6) * 2 * M_PI;
        if (fabs(g_vyaw) >= fabs(g_vx)) {
            pos[0] = (float)(sin(ph) * g_vyaw);
            pos[1] = (float)(sin(ph) * g_vyaw);
        } else {
            pos[0] = (float)(-sin(ph) * g_vx);
            pos[1] = (float)(sin(ph) * g_vx);
        }
        pos[2] = (float)(60.0 * cos(ph));
    } else {
        g_gait_t0 = 0;
    }
    write_pos(pos, 0);
}

void tlp_walk(int vx, int ms) { g_vx = vx; g_vyaw = 0; g_gait_t0 = 0; g_gait_until = ms > 0 ? g_vnow + (int64_t)ms * 1000 : 0; }
void tlp_turn(int vyaw, int ms) { g_vyaw = vyaw; g_vx = 0; g_gait_t0 = 0; g_gait_until = ms > 0 ? g_vnow + (int64_t)ms * 1000 : 0; }
void tlp_gait_stop(void) { g_vx = g_vyaw = 0; g_gait_until = 0; }

/* learned-runtime stubs */
static bool g_rl_cal = false, g_rl_dry = true, g_flip = false, g_swap = false;
bool tlp_rl_calibrate(int step)
{
    char m[96];
    snprintf(m, sizeof m, "[rl] cal step %d done (sim)", step);
    tars_log_shell_say(m);
    if (step == 1) g_rl_cal = true;
    return true;
}
void tlp_rl_mode(bool en, bool dry) { g_rl_on = en; g_rl_dry = dry; }
bool tlp_rl_is_calibrated(void) { return g_rl_cal; }
static int g_sign[3] = {1, 1, 1};
void tlp_rl_map(bool flip, bool swap) { g_flip = flip; g_swap = swap; for (int i = 0; i < 3; i++) g_sign[i] = flip ? -1 : 1; }
void tlp_rl_map3(int a, int b, int c, bool swap) { g_sign[0] = a; g_sign[1] = b; g_sign[2] = c; g_swap = swap; g_flip = (a < 0 && b < 0 && c < 0); }
bool tlp_rl_active(void) { return g_rl_on; }
void tlp_rl_selftest(int step)
{
    if (step == 0) { tars_log_shell_release(); return; }
    const float pose[3] = {step == 1 ? 68.f : 0.f, step == 1 ? 68.f : 0.f, step == 2 ? 90.f : 0.f};
    tars_log_shell_hold_pose(pose, 30000);
}
void tlp_rl_status(char *buf, size_t n)
{
    snprintf(buf, n, "map=%c%c%c%s learned=%s%s cal=%d (sim)", g_sign[0] > 0 ? '+' : '-', g_sign[1] > 0 ? '+' : '-', g_sign[2] > 0 ? '+' : '-',
             g_swap ? "+swapLR" : "", g_rl_on ? "on" : "off", (g_rl_on && g_rl_dry) ? "(dry)" : "", g_rl_cal ? 1 : 0);
}
static int g_rl_auto = 2;                                 /* firmware default: learned mode with walking commands at power-up */
void tlp_rl_autostart(int mode) { g_rl_auto = mode; }
int tlp_rl_autostart_get(void) { return g_rl_auto; }
static int g_gait_set = 6;                                /* firmware default: v9 */
static const char *const k_gait_names[] = {"v4", "v5", "v5, faster forward", "v6, foot plates", "v7, smoother", "v8, smooth fwd+back", "v9, sinusoid hips"};
#define N_GAIT_SETS ((int)(sizeof k_gait_names / sizeof k_gait_names[0]))
static unsigned g_custom_mask = 0;
int tlp_rl_gait_set(int idx, bool save) { (void)save; if (idx < 0 || idx >= N_GAIT_SETS) return -1; if (g_rl_on && g_vx != 0) return -2; g_gait_set = idx; g_custom_mask = 0; return 0; }
int tlp_rl_gait_custom(char which, const int32_t micro[15])
{
    static const char slots[] = "fbtr";
    const char *k = which ? strchr(slots, which) : NULL;
    if (!k) return -1;
    if (micro[10] < 580000 || micro[10] > 2040000) return -1;           /* the firmware checks all 15 parameters (tars_gait_in_range); the stub only the frequency */
    if (g_rl_on && g_vx != 0) return -2;
    g_custom_mask |= 1u << (k - slots);
    return 0;
}
unsigned tlp_rl_gait_custom_mask(void) { return g_custom_mask; }
static int g_getup_mode = 1;                              /* firmware default: learned network first, keyframes on the 2nd attempt */
int tlp_rl_getup_mode(int mode, bool save) { (void)save; if (mode < 0 || mode > 2) return -1; g_getup_mode = mode; return 0; }
int tlp_rl_getup_mode_get(void) { return g_getup_mode; }
int tlp_rl_getup_nn_available(void) { return 1; }
static int g_speed_map = 1, g_speed_max = 90;
void tlp_rl_speed_map(int on) { g_speed_map = on ? 1 : 0; }
int tlp_rl_speed_map_get(void) { return g_speed_map; }
void tlp_rl_speed_max(int pct) { g_speed_max = pct; }
int tlp_rl_speed_max_get(void) { return g_speed_max; }
static int g_straight[4] = {1, 150, 0, 0};               /* on, kp %, kd %, trim per mille */
void tlp_rl_straight(int on, int kp, int kd, int trim) { g_straight[0] = on ? 1 : 0; g_straight[1] = kp; g_straight[2] = kd; g_straight[3] = trim; }
void tlp_rl_straight_get(int out[4]) { memcpy(out, g_straight, sizeof g_straight); }
int tlp_rl_gait_get(void) { return g_gait_set; }
const char *tlp_rl_gait_name(int idx) { return idx >= 0 && idx < N_GAIT_SETS ? k_gait_names[idx] : "?"; }
int tlp_rl_gait_count(void) { return N_GAIT_SETS; }
/* show motions: the stub accepts one at a time while the learned mode is on and never finishes by itself (the host test ends it with `learn motion stop`) */
static const char *const k_motion_names[] = {"nod", "bow", "headbang", "happy", "sad", "angry", "scared", "surprised", "shy", "curious", "sleepy", "proud", "laugh", "loving", "impatient", "wave", "cheer", "windmill", "robotdance", "swim", "twist"};
#define N_MOTIONS ((int)(sizeof k_motion_names / sizeof k_motion_names[0]))
static int g_motion_busy = 0;
int tlp_rl_motion_count(void) { return N_MOTIONS; }
const char *tlp_rl_motion_name(int idx) { return idx >= 0 && idx < N_MOTIONS ? k_motion_names[idx] : "?"; }
int tlp_rl_motion(const char *name, int speed_pct)
{
    int idx = -1;
    for (int i = 0; i < N_MOTIONS; i++) {
        if (!strcmp(name, k_motion_names[i])) idx = i;
    }
    if (idx < 0 || speed_pct < 30 || speed_pct > 100) return -1;
    if (!g_rl_on) return -2;
    if (g_motion_busy) return -3;
    g_motion_busy = 1;
    return 0;
}
void tlp_rl_motion_stop(void) { g_motion_busy = 0; }
static int g_tune[5] = {100, 100, 0, 100, 500};
void tlp_rl_tune(int kp, int kd, int lpf, int amp, int hz) { g_tune[0] = kp; g_tune[1] = kd; g_tune[2] = lpf; g_tune[3] = amp; g_tune[4] = hz; }
void tlp_rl_tune_get(int out[5]) { for (int i = 0; i < 5; i++) out[i] = g_tune[i]; }
bool tlp_rl_history(int idx, char *buf, size_t n)
{
    static const char *const k[] = {"t=1.000s learned mode ON", "t=1.350s STAND -> FALLEN  up=(100,-4,1) attempt 0", "t=2.100s FALLEN -> GETUP1  up=(100,-4,1) attempt 1"};
    if (idx < 0 || idx >= 3) return false;
    snprintf(buf, n, "%s", k[idx]);
    return true;
}
void tlp_info(char *buf, size_t n) { snprintf(buf, n, "build=\"sim\" link=%s baud=%u%s up_ms=%d", g_link_console ? "console" : "sim", (unsigned)g_baud, g_baud_confirmed ? "" : "(unconfirmed)", (int)(g_vnow / 1000)); }
void tlp_error_log(void (*output)(const char *, ...)) { output("@err errors: no flash on simulator"); }
void tlp_reboot(void) { exit(0); }

/* ------------------------------------------------------------------------------------------------ main loop */
int main(int argc, char **argv)
{
    int use_pty = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pty")) use_pty = 1;
    }
    if (getenv("SIM_SPEED")) g_speed = atof(getenv("SIM_SPEED"));
    if (getenv("SIM_RING")) g_ring_cap = (uint32_t)atoi(getenv("SIM_RING"));
    if (getenv("SIM_DEAD")) g_dead = atoi(getenv("SIM_DEAD"));
    if (getenv("SIM_CALIB")) g_calib = atoi(getenv("SIM_CALIB"));
    if (getenv("SIM_OUT_FAIL_AFTER")) g_out_fail_after = atol(getenv("SIM_OUT_FAIL_AFTER"));
    if (getenv("SIM_OUT_STALL_MS")) g_stall_ms = atol(getenv("SIM_OUT_STALL_MS"));
    clock_gettime(CLOCK_MONOTONIC, &g_t0);

    if (use_pty) {
        int m = posix_openpt(O_RDWR | O_NOCTTY);
        if (m < 0 || grantpt(m) || unlockpt(m)) {
            perror("pty");
            return 1;
        }
        struct termios tio;
        int s = open(ptsname(m), O_RDWR | O_NOCTTY);      /* keep a slave open so the master never sees hang-up */
        tcgetattr(s, &tio);
        cfmakeraw(&tio);
        tcsetattr(s, TCSANOW, &tio);
        fprintf(stderr, "PTY=%s\n", ptsname(m));
        fflush(stderr);
        g_in_fd = g_out_fd = m;
    }
    fcntl(g_in_fd, F_SETFL, fcntl(g_in_fd, F_GETFL) | O_NONBLOCK);

    for (int i = 0; i < 3; i++) {
        sv[i].pos = sv[i].des = kZero[i];
    }
    tars_log_shell_init();
    tars_log_shell_say("sim ready");

    int64_t next_servo = 0, next_tick = 0, next_imu = 0;
    int64_t last_real = real_us();
    double vfrac = 0;
    for (;;) {
        /* advance virtual time */
        const int64_t r = real_us();
        vfrac += (double)(r - last_real) * g_speed;
        last_real = r;
        g_vnow += (int64_t)vfrac;
        vfrac -= (double)(int64_t)vfrac;

        /* host -> device */
        uint8_t buf[64];
        ssize_t n = read(g_in_fd, buf, sizeof buf);
        if (n > 0) tars_log_shell_feed(buf, (size_t)n);
        else if (n == 0 && !use_pty && g_in_fd == 0) {
            /* stdin closed: keep running a little to flush a pending session, then leave */
            static int64_t eof_t = 0;
            if (!eof_t) eof_t = g_vnow;
            if (g_vnow - eof_t > 8000000) return 0;
        }

        tars_log_shell_poll();

        while (g_vnow >= next_servo) {
            servo_step(0.001);
            next_servo += 1000;
        }
        /* pending read replies */
        for (int i = 0; i < npend;) {
            if (pend[i].due <= g_vnow) {
                if (pend[i].kind == 0) {
                    if (pend[i].id != g_dead) servo_reply(pend[i].id);
                } else if (g_dead != 1) {
                    tars_log_shell_on_volt(742, g_vnow);
                }
                pend[i] = pend[--npend];
            } else {
                i++;
            }
        }
        if (g_vnow >= next_tick) {
            next_tick = g_vnow + 3000;                                /* robot_task: ~3 ms */
            if (g_calib) tars_log_shell_abort_servo_use();
            if (!tars_log_shell_owns_servos() && !g_calib) classic_gait();
            if (tars_log_shell_bus_active()) {
                tars_log_tick_out_t o;
                tars_log_shell_tick(g_vnow, &o);
                if (o.write && !g_calib) write_pos(o.pos, o.cmd_src);
                if (o.enable_id) sv[o.enable_id - 1].limp = (o.enable_val == 0);
                if (o.read_id && npend < 64) pend[npend++] = (pending_t){g_vnow + 700, 0, o.read_id};
                if (o.read_volt && npend < 64) pend[npend++] = (pending_t){g_vnow + 700, 1, 1};
            }
        }
        if (g_imu_on && g_vnow >= next_imu) {
            next_imu = g_vnow + 10000;
            imu_sample();
        }
        usleep(150);
    }
}
