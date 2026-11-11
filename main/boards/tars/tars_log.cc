/*
 * tars_log.cc - ESP32-S3 glue of the Tars serial log / test tool (see tars_log.h).
 *
 * Serial link: depending on the board the PC cable ends either in the native USB-Serial/JTAG port or - as on the Tars boards seen
 * so far - in a USB-UART bridge on UART0.  The stock console is UART0 (primary, 115200) + USB-Serial/JTAG (secondary, OUTPUT ONLY),
 * so this file polls the receive FIFOs of both ports itself and answers on the port the last command came from:
 *   - native USB : direct writes into the USB transmit FIFO (full USB speed)
 *   - UART0      : direct writes into the UART0 transmit FIFO without busy-waiting (the console would spin), optionally at a
 *                  higher baud rate ('baud 921600', reverts by itself if the PC does not follow)
 *   - console    : plain stdout (reaches both ports; slow; fallback and the answer path before the first command)
 */
#include "tars_log.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <esp_heap_caps.h>
#include <esp_camera.h>
#include <esp_rom_sys.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <driver/i2c_master.h>
#include <driver/uart.h>
#include <hal/uart_ll.h>
#include <hal/usb_serial_jtag_ll.h>

#include "imu.h"
#include "motor.h"
#include "robot.h"
#include "robot_action.h"
#include "tars_log_core.h"
#include "tars_log_port.h"
#include "tars_log_shell.h"
#include "tars_rl.h"

/* driver/usb_serial_jtag.h (component esp_driver_usb_serial_jtag, always linked while the USB console is enabled): true while a USB host is attached */
extern "C" bool usb_serial_jtag_is_connected(void);

namespace {

enum Link { kConsole = 0, kUsj = 1, kUart0 = 2 };

volatile int s_link = kConsole;           /* port the last command arrived on; replies and streams go there */
volatile bool s_link_forced = false;      /* 'link console': always answer through the console (stdout) path */
uint32_t s_baud = 115200;                 /* current UART0 baud rate (changed by the 'baud' command) */
volatile bool s_baud_confirmed = true;    /* false while a rate change waits for the PC to confirm with 'baud ok' */
int64_t s_baud_t0_us = 0;
int64_t s_last_rx_us = 0;
bool s_usj_stalled = false;               /* the USB host does not drain our FIFO: drop instead of waiting */
portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
QueueHandle_t s_msgq = nullptr;
volatile int s_cmd_src = 0;               /* tag of the 'C' events written by the tool itself */
constexpr size_t kMsgChars = 160;

int16_t to_i16(float v)
{
    const long r = lroundf(v);
    return (int16_t)(r > 32767 ? 32767 : (r < -32768 ? -32768 : r));
}

int8_t to_i8(float v)
{
    const long r = lroundf(v);
    return (int8_t)(r > 127 ? 127 : (r < -128 ? -128 : r));
}

/* ---- USB-Serial/JTAG ---------------------------------------------------------------------------------------- */
size_t usj_read(uint8_t *buf, size_t cap)
{
    size_t n = 0;
    while (n < cap && usb_serial_jtag_ll_rxfifo_data_available()) {
        const uint32_t got = usb_serial_jtag_ll_read_rxfifo(buf + n, (uint32_t)(cap - n));
        if (got == 0) break;
        n += got;
    }
    return n;
}

/* Write bytes into the USB transmit FIFO (64 bytes per packet). Returns how many were accepted: less than n means the
 * host is not reading (it will be accepted again as soon as the host drains the FIFO). */
size_t usj_write(const uint8_t *p, size_t n)
{
    if (n == 0) return 0;
    flockfile(stdout);                    /* keep ESP_LOG / printf of other tasks from landing in the middle of our lines */
    size_t off = 0;
    int64_t last_progress = esp_timer_get_time();
    int spins = 0;
    while (off < n) {
        if (usb_serial_jtag_ll_txfifo_writable()) {
            const uint32_t w = usb_serial_jtag_ll_write_txfifo(p + off, (uint32_t)(n - off));
            off += w;
            last_progress = esp_timer_get_time();
            s_usj_stalled = false;
            spins = 0;
            continue;
        }
        if (s_usj_stalled || esp_timer_get_time() - last_progress > 40000) {   /* host gone / not reading */
            s_usj_stalled = true;
            break;
        }
        if (++spins < 400) {
            esp_rom_delay_us(5);          /* the host empties a packet about every 100 us .. 1 ms */
        } else {
            spins = 0;
            vTaskDelay(1);                /* let lower priority tasks (idle / watchdog) run */
        }
    }
    if (off > 0) {
        usb_serial_jtag_ll_txfifo_flush();                    /* hand a partial packet to the host */
        if (off == n && (n % 64) == 0) {                      /* a packet that filled the FIFO exactly needs a zero-length packet after it */
            const int64_t t0 = esp_timer_get_time();
            while (!usb_serial_jtag_ll_txfifo_writable() && esp_timer_get_time() - t0 < 5000) esp_rom_delay_us(10);
            if (usb_serial_jtag_ll_txfifo_writable()) usb_serial_jtag_ll_txfifo_flush();
        }
    }
    funlockfile(stdout);
    return off;
}

/* ---- UART0 ----------------------------------------------------------------------------------------------------- */
size_t uart0_read(uint8_t *buf, size_t cap)
{
    uart_dev_t *hw = UART_LL_GET_HW(0);
    uint32_t avail = uart_ll_get_rxfifo_len(hw);
    if (avail > cap) avail = (uint32_t)cap;
    if (avail) uart_ll_read_rxfifo(hw, buf, avail);
    return avail;
}

/* end of the slice [from, to): cut after the last '\n' inside it so that another task's console line never lands in the
 * middle of one of our lines (a line longer than the slice is cut at `to`) */
size_t line_end_within(const uint8_t *p, size_t from, size_t to)
{
    for (size_t e = to; e > from; e--) {
        if (p[e - 1] == '\n') return e;
    }
    return to;
}

/* UART0 transmit FIFO (128 bytes) without the console's busy-wait: sleep a tick while it is full. The stdout lock is held
 * per slice only, so printf / ESP_LOG of other tasks interleave between slices, never inside one. */
size_t uart0_write(const uint8_t *p, size_t n)
{
    uart_dev_t *hw = UART_LL_GET_HW(0);
    size_t off = 0;
    int64_t last_progress = esp_timer_get_time();
    while (off < n) {
        size_t slice_end = off + ((n - off) > 128 ? 128 : (n - off));
        slice_end = line_end_within(p, off, slice_end);
        flockfile(stdout);
        while (off < slice_end) {
            const uint32_t room = uart_ll_get_txfifo_len(hw);
            if (room >= 8 || (room > 0 && slice_end - off < 8)) {
                const size_t k = (slice_end - off) < room ? (slice_end - off) : room;
                uart_ll_write_txfifo(hw, p + off, (uint32_t)k);
                off += k;
                last_progress = esp_timer_get_time();
            } else if (esp_timer_get_time() - last_progress > 500000) {            /* line stuck: give up */
                funlockfile(stdout);
                return off;
            } else {
                vTaskDelay(1);
            }
        }
        funlockfile(stdout);
    }
    return off;
}

/* normal stdout: reaches UART0 and the USB console, '\n' becomes "\r\n"; spins at the console baud rate, so in slices */
size_t console_write(const uint8_t *p, size_t n)
{
    size_t off = 0;
    while (off < n) {
        const size_t k = line_end_within(p, off, off + ((n - off) > 96 ? 96 : (n - off))) - off;
        const size_t w = fwrite(p + off, 1, k, stdout);
        fflush(stdout);
        off += w;
        if (w < k) break;
    }
    return off;
}

/* change the UART0 rate; whatever was queued (the "@ok baud" reply) leaves at the old rate first */
bool apply_baud(uint32_t baud)
{
    uart_dev_t *hw = UART_LL_GET_HW(0);
    fflush(stdout);
    const int64_t t0 = esp_timer_get_time();
    while (!uart_ll_is_tx_idle(hw) && esp_timer_get_time() - t0 < 300000) esp_rom_delay_us(100);
    if (uart_set_baudrate(UART_NUM_0, baud) != ESP_OK) return false;
    s_baud = baud;
    return true;
}

/* ---- QMI8658 register access for the 'imu' diagnostics: our own device handle on the IMU I2C bus (the stock driver in
 * boards/common/imu.cc keeps its handle private); the I2C master driver serialises concurrent users of the bus ---- */
i2c_master_dev_handle_t s_imu_dev = nullptr;

bool imu_dev_open()
{
    if (s_imu_dev) return true;
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(I2C_NUM_0, &bus) != ESP_OK || !bus) return false;
    static const uint16_t kAddrs[] = {0x6B, 0x6A};
    for (uint16_t addr : kAddrs) {
        if (i2c_master_probe(bus, addr, 50) != ESP_OK) continue;
        i2c_device_config_t cfg = {};
        cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
        cfg.device_address = addr;
        cfg.scl_speed_hz = 400000;
        if (i2c_master_bus_add_device(bus, &cfg, &s_imu_dev) == ESP_OK) return true;
    }
    return false;
}

bool has_eol(const uint8_t *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (b[i] == '\n' || b[i] == '\r') return true;
    }
    return false;
}

void flush_messages()
{
    char msg[kMsgChars];
    while (s_msgq && xQueueReceive(s_msgq, msg, 0) == pdTRUE) tars_log_shell_say(msg);
}

void log_task(void *)
{
    uint8_t buf[64];
    for (;;) {
        size_t n = usb_serial_jtag_is_connected() ? usj_read(buf, sizeof buf) : 0;   /* no USB host (UART-bridge boards): ignore the USB port */
        if (n) {
            s_last_rx_us = esp_timer_get_time();
            if (has_eol(buf, n) && !s_link_forced) s_link = kUsj;
            tars_log_shell_feed(buf, n);
        }
        n = uart0_read(buf, sizeof buf);
        if (n) {
            s_last_rx_us = esp_timer_get_time();
            if (has_eol(buf, n) && !s_link_forced) s_link = kUart0;
            tars_log_shell_feed(buf, n);
        }
        flush_messages();
        const bool more = tars_log_shell_poll();
        if (s_baud != 115200) {                           /* never leave the console at a rate nobody listens to */
            const int64_t now = esp_timer_get_time();
            if ((!s_baud_confirmed && now - s_baud_t0_us > 8000000) ||
                (s_baud_confirmed && tars_log_shell_idle() && now - s_last_rx_us > 30000000)) {
                apply_baud(115200);
                s_baud_confirmed = true;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(more ? 1 : 5));
    }
}

}  // namespace

/* ---------------------------------------------------------------------------------------------------------------
 * platform services for the portable shell
 * --------------------------------------------------------------------------------------------------------------- */
extern "C" {

int64_t tlp_now_us(void) { return esp_timer_get_time(); }
void tlp_delay_ms(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms) ? pdMS_TO_TICKS(ms) : 1); }
void tlp_lock(void) { portENTER_CRITICAL(&s_mux); }
void tlp_unlock(void) { portEXIT_CRITICAL(&s_mux); }

size_t tlp_write(const char *s, size_t n)
{
    switch (s_link) {
    case kUsj: return usj_write((const uint8_t *)s, n);
    case kUart0: return uart0_write((const uint8_t *)s, n);
    default: return console_write((const uint8_t *)s, n);
    }
}

bool tlp_out(const char *s, size_t n) { return tlp_write(s, n) == n; }

void tlp_flush_msgs(void) { flush_messages(); }

void tlp_set_link(int console)
{
    s_link_forced = console != 0;
    s_link = kConsole;                                    /* the next command re-selects the direct path (unless forced) */
}

bool tlp_set_baud(uint32_t baud)
{
    if (!apply_baud(baud)) return false;
    s_baud_confirmed = (baud == 115200);                  /* the PC must follow and send 'baud ok', else we go back */
    s_baud_t0_us = esp_timer_get_time();
    s_last_rx_us = s_baud_t0_us;
    return true;
}

void tlp_baud_confirm(void)
{
    s_baud_confirmed = true;
    s_last_rx_us = esp_timer_get_time();
}

tars_log_ev_t *tlp_alloc_events(uint32_t *cap)
{
    static const uint32_t kPsram[] = {65536, 32768, 16384, 8192};
    for (uint32_t c : kPsram) {
        void *p = heap_caps_malloc((size_t)c * sizeof(tars_log_ev_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (p) {
            *cap = c;
            return (tars_log_ev_t *)p;
        }
    }
    static const uint32_t kInternal[] = {1536, 768, 384};
    for (uint32_t c : kInternal) {
        void *p = heap_caps_malloc((size_t)c * sizeof(tars_log_ev_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (p) {
            *cap = c;
            return (tars_log_ev_t *)p;
        }
    }
    return nullptr;
}

bool tlp_imu_read(uint8_t reg, uint8_t *buf, size_t n)
{
    return imu_dev_open() && i2c_master_transmit_receive(s_imu_dev, &reg, 1, buf, n, 50) == ESP_OK;
}

bool tlp_imu_write(uint8_t reg, uint8_t val)
{
    const uint8_t data[2] = {reg, val};
    return imu_dev_open() && i2c_master_transmit(s_imu_dev, data, sizeof data, 50) == ESP_OK;
}

void tlp_imu_stream(bool on) { tars_rl_imu_stream(on); }
bool tlp_imu_ok(void) { return imu_is_initialized(); }
int tlp_calibrating(void) { return calibrate_mode; }
int16_t tlp_zero_pos(int idx) { return (idx >= 0 && idx < MOTOR_NUM) ? motor[idx].ZeroPos : 0; }
int tlp_servo_mask(int idx) { return MotorMask(idx); }
void tlp_walk(int vx_counts, int ms) { tars_set_forward(vx_counts, ms); }
void tlp_turn(int vyaw_counts, int ms) { tars_set_turn(vyaw_counts, ms); }
void tlp_gait_stop(void) { tars_set_forward(0, 0); }

bool tlp_rl_calibrate(int step) { return tars_rl_calibrate_imu(step); }
void tlp_rl_mode(bool enable, bool dry_run) { tars_rl_set_mode(enable, dry_run); }
bool tlp_rl_is_calibrated(void) { return tars_rl_is_calibrated(); }
void tlp_rl_map(bool flip, bool swap_lr) { tars_rl_set_mapping(flip, swap_lr); }
void tlp_rl_map3(int s1, int s2, int s3, bool swap_lr) { tars_rl_set_mapping3(s1, s2, s3, swap_lr); }
bool tlp_rl_active(void) { return tars_rl_active(); }
void tlp_rl_selftest(int step) { tars_rl_selftest(step); }
void tlp_rl_status(char *buf, size_t n) { tars_rl_status(buf, n); }
void tlp_rl_autostart(int mode) { tars_rl_set_autostart(mode); }
int tlp_rl_autostart_get(void) { return tars_rl_autostart(); }
bool tlp_rl_history(int idx, char *buf, size_t n) { return tars_rl_history(idx, buf, n); }
int tlp_rl_gait_set(int idx, bool save) { return tars_rl_set_gait_set(idx, save); }
int tlp_rl_gait_get(void) { return tars_rl_gait_set(); }
int tlp_rl_getup_mode(int mode, bool save) { return tars_rl_set_getup_mode(mode, save); }
int tlp_rl_getup_mode_get(void) { return tars_rl_getup_mode(); }
int tlp_rl_getup_nn_available(void) { return tars_rl_getup_nn_available(); }
const char *tlp_rl_gait_name(int idx) { return tars_rl_gait_name(idx); }
int tlp_rl_gait_count(void) { return tars_rl_gait_count(); }
int tlp_rl_gait_custom(char which, const int32_t micro[15])
{
    float p[15];
    for (int i = 0; i < 15; i++) p[i] = (float)micro[i] * 1e-6f;
    return tars_rl_set_custom_gait(which, p);
}
unsigned tlp_rl_gait_custom_mask(void) { return tars_rl_gait_custom_mask(); }
void tlp_rl_speed_map(int on) { tars_rl_set_speed_map(on); }
int tlp_rl_speed_map_get(void) { return tars_rl_speed_map(); }
void tlp_rl_speed_max(int pct) { tars_rl_set_speed_max(pct); }
int tlp_rl_speed_max_get(void) { return tars_rl_speed_max(); }
void tlp_rl_straight(int on, int kp_pct, int kd_pct, int trim_pm) { tars_rl_set_straight(on, kp_pct, kd_pct, trim_pm); }
void tlp_rl_straight_get(int out[4]) { tars_rl_get_straight(out); }
void tlp_rl_tune(int kp_pct, int kd_pct, int lpf_hz, int amp_pct, int rate_hz) { tars_rl_set_tune(kp_pct, kd_pct, lpf_hz, amp_pct, rate_hz); }
void tlp_rl_tune_get(int out[5]) { tars_rl_get_tune(out); }
int tlp_rl_motion_count(void) { return tars_rl_motion_count(); }
const char *tlp_rl_motion_name(int idx) { return tars_rl_motion_name(idx); }
int tlp_rl_motion(const char *name, int speed_pct) { return tars_rl_motion(tars_rl_motion_find(name), speed_pct); }
void tlp_rl_motion_stop(void) { tars_rl_motion_stop(); }

void tlp_info(char *buf, size_t n)
{
    constexpr uint32_t dma_caps = MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL;
    const char* camera_dma = "off";
    if (esp_camera_sensor_get() != nullptr) {
        camera_dma = esp_camera_get_psram_mode() ? "psram" : "internal";
    }
    snprintf(buf, n, "build=\"%s %s\" link=%s baud=%u up_ms=%u heap=%u psram=%u imu=%d rst=%s dma=%u dma_max=%u cam_dma=%s", __DATE__, __TIME__,
             s_link_forced ? "console" : (s_link == kUsj ? "usb" : (s_link == kUart0 ? "uart0" : "console")), (unsigned)s_baud,
             (unsigned)(esp_timer_get_time() / 1000), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), imu_is_initialized() ? 1 : 0, tars_reset_reason_name(),
             (unsigned)heap_caps_get_free_size(dma_caps), (unsigned)heap_caps_get_largest_free_block(dma_caps), camera_dma);
}

void tlp_reboot(void) { esp_restart(); }

/* ---------------------------------------------------------------------------------------------------------------
 * public API
 * --------------------------------------------------------------------------------------------------------------- */
void tars_log_init(void)
{
    tars_log_shell_init();
    s_msgq = xQueueCreate(8, kMsgChars);
    xTaskCreatePinnedToCore(log_task, "tars_log", 8192, nullptr, 2, nullptr, 1);
    printf("tars_log: serial log/test tool ready (PC side: python3 tools/tars_log/tars_serial.py check)\n");
}

bool tars_log_owns_bus(void) { return tars_log_shell_bus_active(); }
bool tars_log_owns_servos(void) { return tars_log_shell_owns_servos(); }

void tars_log_bus_tick(void)
{
    if (calibrate_mode != 0) tars_log_shell_abort_servo_use();     /* servo zero calibration started: the tool stands down */
    tars_log_tick_out_t o;
    tars_log_shell_tick(esp_timer_get_time(), &o);
    if (o.write && calibrate_mode == 0) {
        s_cmd_src = o.cmd_src;
        WritePos(o.pos);
        s_cmd_src = 0;
    }
    if (o.enable_id) EnableMotor((uint8_t)o.enable_id, (uint8_t)o.enable_val);
    if (o.read_id) ReadMotorState((uint8_t)o.read_id);
    if (o.read_volt) ReadServoVoltage(1);
}

void tars_log_sample(const tars_log_imu_sample_t *s)
{
    tars_log_imu_t t;
    for (int i = 0; i < 3; i++) {
        t.acc[i] = to_i16(s->acc[i] * (16384.0f / 9.8f));
        t.gyro[i] = to_i16(s->gyro[i] * 32.0f);
        t.up[i] = to_i16(s->up[i] * 10000.0f);
    }
    t.flags = s->flags;
    t.state = s->state;
    t.vx = to_i8(vx);
    t.vyaw = to_i8(vyaw);
    tars_log_shell_on_imu(&t, esp_timer_get_time());
}

void tars_log_servo_fb(int id, int pos, int spd, int load)
{
    tars_log_shell_on_servo(id, (int16_t)pos, (int16_t)spd, (int16_t)load, esp_timer_get_time());
}

void tars_log_servo_volt(float volts)
{
    tars_log_shell_on_volt(to_i16(volts * 100.0f), esp_timer_get_time());
}

void tars_log_servo_cmd(const short pos[3], short speed)
{
    tars_log_shell_on_cmd(pos, speed, s_cmd_src, esp_timer_get_time());
}

void tars_log_printf(const char *fmt, ...)
{
    char msg[kMsgChars];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    tars_log_shell_on_msg(msg, esp_timer_get_time());
    if (s_msgq) xQueueSend(s_msgq, msg, 0);          /* printed by the log task; dropped when the queue is full */
}

void tars_log_hold_pose(const float pos_counts[3], unsigned timeout_ms)
{
    tars_log_shell_hold_pose(pos_counts, timeout_ms);
}

void tars_log_release_servos(void) { tars_log_shell_release(); }

}  // extern "C"
