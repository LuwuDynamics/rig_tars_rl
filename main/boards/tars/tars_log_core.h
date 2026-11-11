/*
 * tars_log_core.h - portable C99 building blocks of the Tars serial log / test tool.
 *
 * No ESP-IDF dependency: the same file is compiled for the robot and for the host-side self test
 * (tools/tars_log/test/).  Contents:
 *   - a wait-free multi-producer event ring (seqlock slots), used to record IMU / servo / command samples
 *   - integer-only text formatting of events ("@I,t_us,...*HH") - this build uses newlib-nano printf (no %f, no %lld)
 *   - tiny parsing helpers for the command line
 *   - servo test waveforms (step / sine)
 */
#ifndef TARS_LOG_CORE_H
#define TARS_LOG_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------------------------------------------
 * events
 * --------------------------------------------------------------------------------------------------------------- */
#define TLE_IMU   'I'   /* v: ax ay az gx gy gz ux uy uz flags state vxy                                  */
#define TLE_SERVO 'S'   /* id = servo 1..3; v: pos spd load lat_us                                        */
#define TLE_CMD   'C'   /* id = source tag; v: pos1 pos2 pos3 speed (servo register units)                */
#define TLE_VOLT  'V'   /* v: centivolt                                                                   */
#define TLE_MSG   'M'   /* v: up to 24 characters of text                                                 */

#define TARS_LOG_EV_VALS 12
#define TARS_LOG_MSG_CHARS (TARS_LOG_EV_VALS * 2)

typedef struct {
    uint32_t seq;                    /* ring index + 1 once the slot is committed, 0 while empty / being written */
    int32_t t_us;                    /* microseconds since the recording epoch                                    */
    uint8_t type;
    uint8_t id;
    int16_t v[TARS_LOG_EV_VALS];
    uint8_t pad[2];
} tars_log_ev_t;                     /* 36 bytes */

typedef struct {
    tars_log_ev_t *slots;
    uint32_t cap;
    uint32_t head;                   /* next index to reserve (atomic, monotonic)                                  */
    uint32_t dropped;                /* events lost because a non-overwrite ring was full                          */
    int64_t epoch_us;
    bool overwrite;                  /* true = circular (live streaming), false = stop when full (recording)       */
} tars_log_ring_t;

void tars_log_ring_init(tars_log_ring_t *r, tars_log_ev_t *mem, uint32_t cap);
/* NOT thread-safe: call only while no producer is active (recording flag off). Clears the slots that were used. */
void tars_log_ring_reset(tars_log_ring_t *r, bool overwrite, int64_t epoch_us);
/* wait-free, any task; returns false if the event was dropped (ring full) */
bool tars_log_ring_push(tars_log_ring_t *r, int64_t now_us, uint8_t type, uint8_t id, const int16_t *v, int nv);
/* number of events stored (non-overwrite) / index range available is [tars_log_ring_first, head) */
uint32_t tars_log_ring_first(const tars_log_ring_t *r);
uint32_t tars_log_ring_head(const tars_log_ring_t *r);
/* 1 = copied, 0 = slot not committed yet (producer still writing), -1 = already overwritten / torn */
int tars_log_ring_get(const tars_log_ring_t *r, uint32_t idx, tars_log_ev_t *out);

/* ---------------------------------------------------------------------------------------------------------------
 * text formatting (integer only)
 * --------------------------------------------------------------------------------------------------------------- */
/* "@I,12345,..*HH\n".  Returns the line length (incl. '\n'), 0 if `cap` is too small or the type is unknown. */
size_t tars_log_format_event(const tars_log_ev_t *e, char *out, size_t cap);
/* append "*HH" + '\n' to a line that starts with '@' and has no checksum yet (length n, buffer size cap). Returns new length or 0. */
size_t tars_log_seal_line(char *line, size_t n, size_t cap);
/* verify a received line "@...*HH" (without the trailing newline); returns true if the checksum matches */
bool tars_log_check_line(const char *line);
uint8_t tars_log_checksum(const char *s, size_t n);

/* fixed-point helpers: write `milli` (value*1000) as "-12.345" with `decimals` (0..3) digits; returns chars written (no NUL) */
size_t tars_log_put_milli(char *out, int32_t milli, int decimals);
size_t tars_log_put_i32(char *out, int32_t v);

/* ---------------------------------------------------------------------------------------------------------------
 * command-line parsing
 * --------------------------------------------------------------------------------------------------------------- */
/* split on blanks, in place; returns argc (<= max_argv) */
int tars_log_tokenize(char *line, char **argv, int max_argv);
bool tars_log_parse_i32(const char *s, int32_t *out);       /* decimal integer, optional sign, no trailing characters */
bool tars_log_parse_milli(const char *s, int32_t *out);     /* "1.5" -> 1500, "-0.25" -> -250, "3" -> 3000           */

/* ---------------------------------------------------------------------------------------------------------------
 * servo test waveforms. Positions are in WritePos() units: offset in servo counts from the calibrated zero.
 * --------------------------------------------------------------------------------------------------------------- */
#define TLW_NONE 0
#define TLW_STEP 1   /* 0 -> +amp -> 0 -> -amp -> 0, `hold_ms` per level */
#define TLW_SINE 2   /* amp * sin(2 pi f t) for `cycles` periods */

typedef struct {
    uint8_t kind;
    uint8_t id;            /* servo 1..3 */
    int16_t amp;           /* counts */
    uint16_t hold_ms;      /* STEP: dwell per level */
    uint16_t cycles;       /* SINE */
    uint32_t freq_mhz;     /* SINE: milli-Hz */
    uint16_t lead_ms;      /* quiet time at 0 before the motion starts */
    uint16_t tail_ms;      /* quiet time at 0 after it ends */
} tars_log_wave_t;

int32_t tars_log_wave_duration_ms(const tars_log_wave_t *w);
/* positions for time t_ms since the start of the test; returns false once t_ms is past the end (pos = 0) */
bool tars_log_wave_eval(const tars_log_wave_t *w, int32_t t_ms, float pos[3]);

/* ---------------------------------------------------------------------------------------------------------------
 * 'C' (command) event decimation: log when >= 10 ms passed, or >= 1 ms passed and a servo target moved >= 8 counts
 * --------------------------------------------------------------------------------------------------------------- */
typedef struct {
    int64_t last_us;
    int16_t last[3];
    bool have;
} tars_log_cmd_filter_t;
bool tars_log_cmd_should_log(tars_log_cmd_filter_t *f, int64_t now_us, const int16_t pos[3]);

#ifdef __cplusplus
}
#endif
#endif
