/*
 * tars_log_core.c - portable C99 building blocks of the Tars serial log / test tool (see tars_log_core.h).
 */
#include "tars_log_core.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------------------------------------------------
 * ring buffer: wait-free multi-producer, single-consumer; every slot is a small seqlock
 *   writer: seq = 0 ; fence ; payload ; seq = idx + 1 (release)
 *   reader: s1 = seq (acquire) ; copy ; fence ; s2 = seq ; valid when s1 == s2 == idx + 1
 * --------------------------------------------------------------------------------------------------------------- */
void tars_log_ring_init(tars_log_ring_t *r, tars_log_ev_t *mem, uint32_t cap)
{
    r->slots = mem;
    r->cap = cap;
    r->head = 0;
    r->dropped = 0;
    r->epoch_us = 0;
    r->overwrite = false;
    memset(mem, 0, (size_t)cap * sizeof(tars_log_ev_t));
}

void tars_log_ring_reset(tars_log_ring_t *r, bool overwrite, int64_t epoch_us)
{
    const uint32_t used = r->head < r->cap ? r->head : r->cap;
    memset(r->slots, 0, (size_t)used * sizeof(tars_log_ev_t));
    r->dropped = 0;
    r->overwrite = overwrite;
    r->epoch_us = epoch_us;
    __atomic_store_n(&r->head, 0u, __ATOMIC_RELEASE);
}

bool tars_log_ring_push(tars_log_ring_t *r, int64_t now_us, uint8_t type, uint8_t id, const int16_t *v, int nv)
{
    const uint32_t idx = __atomic_fetch_add(&r->head, 1u, __ATOMIC_RELAXED);
    if (!r->overwrite && idx >= r->cap) {
        __atomic_fetch_add(&r->dropped, 1u, __ATOMIC_RELAXED);
        return false;
    }
    tars_log_ev_t *s = &r->slots[idx % r->cap];
    __atomic_store_n(&s->seq, 0u, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    int64_t dt = now_us - r->epoch_us;
    if (dt > 2147483647LL) dt = 2147483647LL;
    if (dt < -2147483647LL) dt = -2147483647LL;
    s->t_us = (int32_t)dt;
    s->type = type;
    s->id = id;
    for (int i = 0; i < TARS_LOG_EV_VALS; i++) s->v[i] = (v && i < nv) ? v[i] : 0;
    __atomic_store_n(&s->seq, idx + 1u, __ATOMIC_RELEASE);
    return true;
}

uint32_t tars_log_ring_first(const tars_log_ring_t *r)
{
    const uint32_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    return (r->overwrite && head > r->cap) ? head - r->cap : 0;
}

uint32_t tars_log_ring_head(const tars_log_ring_t *r)
{
    const uint32_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    return (!r->overwrite && head > r->cap) ? r->cap : head;
}

int tars_log_ring_get(const tars_log_ring_t *r, uint32_t idx, tars_log_ev_t *out)
{
    const uint32_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    if (r->overwrite && head > r->cap && idx < head - r->cap) return -1;
    const tars_log_ev_t *s = &r->slots[idx % r->cap];
    const uint32_t s1 = __atomic_load_n(&s->seq, __ATOMIC_ACQUIRE);
    if (s1 != idx + 1u) return (s1 > idx + 1u) ? -1 : 0;
    out->t_us = s->t_us;
    out->type = s->type;
    out->id = s->id;
    memcpy(out->v, s->v, sizeof(out->v));
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const uint32_t s2 = __atomic_load_n(&s->seq, __ATOMIC_RELAXED);
    if (s2 != s1) return -1;
    out->seq = s1;
    return 1;
}

/* ---------------------------------------------------------------------------------------------------------------
 * integer-only text formatting
 * --------------------------------------------------------------------------------------------------------------- */
size_t tars_log_put_i32(char *out, int32_t v)
{
    char tmp[12];
    int n = 0;
    uint32_t u = (v < 0) ? (uint32_t)0 - (uint32_t)v : (uint32_t)v;
    do {
        tmp[n++] = (char)('0' + (u % 10u));
        u /= 10u;
    } while (u);
    size_t k = 0;
    if (v < 0) out[k++] = '-';
    while (n) out[k++] = tmp[--n];
    return k;
}

size_t tars_log_put_milli(char *out, int32_t milli, int decimals)
{
    if (decimals < 0) decimals = 0;
    if (decimals > 3) decimals = 3;
    static const int32_t pow10_[4] = {1, 10, 100, 1000};
    const int32_t drop = pow10_[3 - decimals];                 /* 1000 -> `decimals` digits */
    const bool neg = milli < 0;
    uint32_t a = neg ? (uint32_t)0 - (uint32_t)milli : (uint32_t)milli;
    a = (a + (uint32_t)(drop / 2)) / (uint32_t)drop;           /* round half away from zero */
    const uint32_t unit = (uint32_t)pow10_[decimals];
    const uint32_t ip = a / unit, fp = a % unit;
    size_t k = 0;
    if (neg && a != 0) out[k++] = '-';
    k += tars_log_put_i32(out + k, (int32_t)ip);
    if (decimals > 0) {
        out[k++] = '.';
        for (int d = decimals - 1; d >= 0; d--) {
            out[k++] = (char)('0' + (fp / (uint32_t)pow10_[d]) % 10u);
        }
    }
    return k;
}

uint8_t tars_log_checksum(const char *s, size_t n)
{
    uint8_t c = 0;
    for (size_t i = 0; i < n; i++) c ^= (uint8_t)s[i];
    return c;
}

size_t tars_log_seal_line(char *line, size_t n, size_t cap)
{
    if (n < 2 || line[0] != '@' || n + 4 > cap) return 0;
    static const char hex[] = "0123456789ABCDEF";
    const uint8_t c = tars_log_checksum(line + 1, n - 1);
    line[n++] = '*';
    line[n++] = hex[c >> 4];
    line[n++] = hex[c & 15];
    line[n++] = '\n';
    return n;
}

bool tars_log_check_line(const char *line)
{
    if (!line || line[0] != '@') return false;
    const char *star = strrchr(line, '*');
    if (!star || star == line) return false;
    int v[2];
    for (int i = 0; i < 2; i++) {
        const char ch = star[1 + i];
        if (ch >= '0' && ch <= '9') v[i] = ch - '0';
        else if (ch >= 'A' && ch <= 'F') v[i] = ch - 'A' + 10;
        else if (ch >= 'a' && ch <= 'f') v[i] = ch - 'a' + 10;
        else return false;
    }
    const char tail = star[3];
    if (tail != '\0' && tail != '\r' && tail != '\n') return false;
    return tars_log_checksum(line + 1, (size_t)(star - line - 1)) == (uint8_t)(v[0] * 16 + v[1]);
}

size_t tars_log_format_event(const tars_log_ev_t *e, char *out, size_t cap)
{
    if (cap < 140) return 0;
    char *p = out;
    int first = 0, count = 0;
    bool with_id = false;
    switch (e->type) {
    case TLE_IMU:   first = 0; count = 12; break;
    case TLE_SERVO: with_id = true; first = 0; count = 4; break;
    case TLE_CMD:   with_id = true; first = 0; count = 4; break;
    case TLE_VOLT:  first = 0; count = 1; break;
    case TLE_MSG:   break;
    default: return 0;
    }
    *p++ = '@';
    *p++ = (char)e->type;
    *p++ = ',';
    p += tars_log_put_i32(p, e->t_us);
    if (with_id) {
        *p++ = ',';
        p += tars_log_put_i32(p, e->id);
    }
    if (e->type == TLE_MSG) {
        *p++ = ',';
        const char *txt = (const char *)e->v;
        for (int i = 0; i < TARS_LOG_MSG_CHARS && txt[i]; i++) {
            const char ch = txt[i];
            *p++ = (ch >= 32 && ch < 127 && ch != '*') ? ch : ' ';
        }
    } else {
        for (int i = 0; i < count; i++) {
            *p++ = ',';
            p += tars_log_put_i32(p, e->v[first + i]);
        }
    }
    return tars_log_seal_line(out, (size_t)(p - out), cap);
}

/* ---------------------------------------------------------------------------------------------------------------
 * parsing
 * --------------------------------------------------------------------------------------------------------------- */
int tars_log_tokenize(char *line, char **argv, int max_argv)
{
    int argc = 0;
    char *p = line;
    while (*p && argc < max_argv) {
        while (*p == ' ' || *p == '\t') *p++ = '\0';
        if (!*p) break;
        argv[argc++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
    }
    return argc;
}

bool tars_log_parse_i32(const char *s, int32_t *out)
{
    if (!s || !*s) return false;
    bool neg = false;
    if (*s == '+' || *s == '-') {
        neg = (*s == '-');
        s++;
    }
    if (!*s) return false;
    int64_t v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return false;
        v = v * 10 + (*s - '0');
        if (v > 2147483648LL) return false;
    }
    if (neg) v = -v;
    if (v > 2147483647LL || v < -2147483647LL - 1) return false;
    *out = (int32_t)v;
    return true;
}

bool tars_log_parse_milli(const char *s, int32_t *out)
{
    if (!s || !*s) return false;
    bool neg = false;
    if (*s == '+' || *s == '-') {
        neg = (*s == '-');
        s++;
    }
    if (!*s) return false;
    int64_t ip = 0;
    int nd = 0;
    while (*s >= '0' && *s <= '9') {
        ip = ip * 10 + (*s - '0');
        if (ip > 2000000LL) return false;
        nd++;
        s++;
    }
    int64_t frac = 0;
    int fd = 0;
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            if (fd < 3) {
                frac = frac * 10 + (*s - '0');
                fd++;
            }
            s++;
        }
        nd += fd;
    }
    if (*s != '\0' || nd == 0) return false;
    while (fd < 3) {
        frac *= 10;
        fd++;
    }
    int64_t v = ip * 1000 + frac;
    *out = (int32_t)(neg ? -v : v);
    return true;
}

/* ---------------------------------------------------------------------------------------------------------------
 * waveforms
 * --------------------------------------------------------------------------------------------------------------- */
static int32_t sine_period_ms(const tars_log_wave_t *w)
{
    return w->freq_mhz ? (int32_t)(1000000LL / (int64_t)w->freq_mhz) : 0;
}

int32_t tars_log_wave_duration_ms(const tars_log_wave_t *w)
{
    int64_t body = 0;
    if (w->kind == TLW_STEP) body = 4LL * w->hold_ms;
    else if (w->kind == TLW_SINE) body = (int64_t)sine_period_ms(w) * w->cycles;
    return (int32_t)(w->lead_ms + body + w->tail_ms);
}

bool tars_log_wave_eval(const tars_log_wave_t *w, int32_t t_ms, float pos[3])
{
    pos[0] = pos[1] = pos[2] = 0.0f;
    if (w->kind == TLW_NONE || w->id < 1 || w->id > 3) return false;
    if (t_ms >= tars_log_wave_duration_ms(w)) return false;
    const int32_t u = t_ms - (int32_t)w->lead_ms;
    if (u < 0) return true;
    float val = 0.0f;
    if (w->kind == TLW_STEP) {
        if (w->hold_ms == 0) return true;
        const int32_t seg = u / (int32_t)w->hold_ms;
        if (seg == 0) val = (float)w->amp;
        else if (seg == 2) val = -(float)w->amp;
    } else if (w->kind == TLW_SINE) {
        const int32_t period = sine_period_ms(w);
        if (period > 0 && u < period * (int32_t)w->cycles) {
            const float f_hz = (float)w->freq_mhz * 0.001f;
            val = (float)w->amp * sinf(6.2831853f * f_hz * ((float)u * 0.001f));
        }
    }
    pos[w->id - 1] = val;
    return true;
}

/* ---------------------------------------------------------------------------------------------------------------
 * command decimation
 * --------------------------------------------------------------------------------------------------------------- */
bool tars_log_cmd_should_log(tars_log_cmd_filter_t *f, int64_t now_us, const int16_t pos[3])
{
    bool log = !f->have;
    if (!log) {
        const int64_t dt = now_us - f->last_us;
        int maxd = 0;
        for (int i = 0; i < 3; i++) {
            int d = pos[i] - f->last[i];
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
        }
        log = dt >= 10000 || (dt >= 1000 && maxd >= 8);
    }
    if (log) {
        f->have = true;
        f->last_us = now_us;
        for (int i = 0; i < 3; i++) f->last[i] = pos[i];
    }
    return log;
}
