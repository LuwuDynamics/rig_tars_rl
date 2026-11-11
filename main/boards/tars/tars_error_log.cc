#include "tars_log.h"
#include "tars_log_port.h"
#include "tars_rl.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iterator>

#include <esp_log.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <nvs_flash.h>

namespace {
constexpr char kTag[] = "TarsErrors";
constexpr char kPartition[] = "tars_err";
constexpr size_t kCapacity = 10;
constexpr uint32_t kRepeatWindowMs = 10000;

struct ErrorRecord {
    uint64_t sequence;
    uint32_t uptime_ms;
    uint32_t occurrences;
    char tag[24];
    char text[128];
};

ErrorRecord records[kCapacity]{};
uint64_t last_sequence = 0;
bool recorded_this_boot = false;
uint16_t dirty_slots = 0;
portMUX_TYPE records_lock = portMUX_INITIALIZER_UNLOCKED;
nvs_handle_t storage = 0;
TaskHandle_t writer_task = nullptr;
std::atomic<vprintf_like_t> console_print{&vprintf};

void ReadRecords(ErrorRecord* entries) {
    for (size_t i = 0; i < kCapacity; ++i) {
        const char key[] = {'e', '1', static_cast<char>('0' + i), '\0'};
        size_t size = sizeof(ErrorRecord);
        ErrorRecord entry{};
        if (nvs_get_blob(storage, key, &entry, &size) == ESP_OK && size == sizeof(entry) &&
            entry.sequence != 0 && entry.sequence % kCapacity == i && entry.occurrences != 0 &&
            entry.tag[sizeof(entry.tag) - 1] == '\0' && entry.text[sizeof(entry.text) - 1] == '\0') {
            entries[i] = entry;
        }
    }
}

void QueueRecord(ErrorRecord entry) {
    bool new_error = false;
    portENTER_CRITICAL(&records_lock);
    auto& last = records[last_sequence % kCapacity];
    if (recorded_this_boot && entry.uptime_ms - last.uptime_ms <= kRepeatWindowMs &&
        strcmp(last.tag, entry.tag) == 0 && strcmp(last.text, entry.text) == 0) {
        last.uptime_ms = entry.uptime_ms;
        if (last.occurrences < UINT32_MAX) {
            ++last.occurrences;
        }
    } else {
        entry.sequence = ++last_sequence;
        entry.occurrences = 1;
        records[last_sequence % kCapacity] = entry;
        recorded_this_boot = true;
        new_error = true;
    }
    dirty_slots |= 1U << (last_sequence % kCapacity);
    portEXIT_CRITICAL(&records_lock);
    if (new_error) {
        xTaskNotifyGive(writer_task);
    }
}

void WriteRecords(void*) {
    bool write_failed = false;
    for (;;) {
        // Flush new errors promptly. A repeated error updates its count on the
        // next batch or timeout, so a flood cannot keep rewriting the same slot.
        if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(30000)) != 0) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
        for (size_t i = 0; i < kCapacity; ++i) {
            const uint16_t bit = 1U << i;
            ErrorRecord entry{};
            portENTER_CRITICAL(&records_lock);
            if (dirty_slots & bit) {
                entry = records[i];
                dirty_slots &= ~bit;
            }
            portEXIT_CRITICAL(&records_lock);
            if (!entry.sequence) {
                continue;
            }

            const char key[] = {'e', '1', static_cast<char>('0' + i), '\0'};
            auto err = nvs_set_blob(storage, key, &entry, sizeof(entry));
            if (err == ESP_OK) {
                err = nvs_commit(storage);
            }
            if (err != ESP_OK) {
                portENTER_CRITICAL(&records_lock);
                dirty_slots |= bit;
                portEXIT_CRITICAL(&records_lock);
                if (!write_failed) {
                    ESP_LOGW(kTag, "Could not save error log: %s", esp_err_to_name(err));
                }
                write_failed = true;
                vTaskDelay(pdMS_TO_TICKS(10000));
                break;
            }
            write_failed = false;
        }
    }
}

// Keep numeric error codes, but never persist variable strings from protocols,
// since those can include chat text or credentials. Keep the template instead.
bool NumericFormat(const char* format) {
    for (auto p = strchr(format, '%'); p; p = strchr(p + 1, '%')) {
        if (*++p == '%') {
            continue;
        }
        p += strspn(p, "#0- +.123456789*");
        if (*p == 'h' || *p == 'l') {
            ++p;
        }
        if (*p == '\0' || !strchr("diuoxXfFeEgGaA", *p)) {
            return false;
        }
    }
    return true;
}

#if CONFIG_LOG_VERSION_1 && CONFIG_LOG_TIMESTAMP_SOURCE_RTOS
int CaptureError(const char* format, va_list args) {
    // Log V1 passes one preformatted prefix, followed by the message format.
    constexpr char kErrorPrefix[] = LOG_COLOR_E "E (%" PRIu32 ") %s: ";
    if (strncmp(format, kErrorPrefix, sizeof(kErrorPrefix) - 1) == 0 &&
        !xPortInIsrContext() && xTaskGetCurrentTaskHandle() != writer_task) {
        ErrorRecord entry{};
        va_list copy;
        va_copy(copy, args);
        entry.uptime_ms = va_arg(copy, uint32_t);
        const char* tag = va_arg(copy, const char*);
        snprintf(entry.tag, sizeof(entry.tag), "%s", tag ? tag : "?");
        const char* body = format + sizeof(kErrorPrefix) - 1;
        if (NumericFormat(body)) {
            vsnprintf(entry.text, sizeof(entry.text), body, copy);
        } else {
            snprintf(entry.text, sizeof(entry.text), "%s", body);
        }
        va_end(copy);
        entry.tag[strcspn(entry.tag, "\r\n\033")] = '\0';
        entry.text[strcspn(entry.text, "\r\n\033")] = '\0';
        QueueRecord(entry);
    }
    return console_print.load(std::memory_order_relaxed)(format, args);
}
#endif

void RecordResetReason() {
    const auto reason = esp_reset_reason();
    switch (reason) {
        case ESP_RST_PANIC:
        case ESP_RST_INT_WDT:
        case ESP_RST_TASK_WDT:
        case ESP_RST_WDT:
        case ESP_RST_BROWNOUT:
        case ESP_RST_PWR_GLITCH:
        case ESP_RST_CPU_LOCKUP: {
            ErrorRecord entry{};
            entry.uptime_ms = esp_log_timestamp();
            snprintf(entry.tag, sizeof(entry.tag), "reset");
            snprintf(entry.text, sizeof(entry.text), "Previous reset: %s (%d)", tars_reset_reason_name(), reason);
            QueueRecord(entry);
            break;
        }
        default:
            break;
    }
}
}  // namespace

void tars_error_log_init(void) {
    auto err = nvs_flash_init_partition(kPartition);
    if (err == ESP_OK) {
        err = nvs_open_from_partition(kPartition, "errors", NVS_READWRITE, &storage);
    }
    if (err != ESP_OK) {
        ESP_LOGW(kTag, "Error log unavailable: %s", esp_err_to_name(err));
        return;
    }
    ReadRecords(records);
    for (const auto& entry : records) {
        last_sequence = std::max(last_sequence, entry.sequence);
    }
    if (xTaskCreate(WriteRecords, "tars_errors", 4096, nullptr, 1, &writer_task) != pdPASS) {
        nvs_close(storage);
        storage = 0;
        ESP_LOGW(kTag, "Could not start error log writer");
        return;
    }
#if CONFIG_LOG_VERSION_1 && CONFIG_LOG_TIMESTAMP_SOURCE_RTOS
    console_print.store(esp_log_set_vprintf(CaptureError), std::memory_order_relaxed);
#else
    ESP_LOGW(kTag, "Error capture requires ESP-IDF Log V1 with uptime timestamps");
#endif
    RecordResetReason();
}

void tlp_error_log(void (*output)(const char *, ...)) {
    if (!storage) {
        output("@err errors: log storage unavailable");
        return;
    }
    ErrorRecord saved[kCapacity]{};
    ReadRecords(saved);
    std::sort(std::begin(saved), std::end(saved), [](const ErrorRecord& a, const ErrorRecord& b) {
        return a.sequence < b.sequence;
    });
    unsigned count = 0;
    for (const auto& entry : saved) {
        if (entry.sequence) {
            // This firmware uses newlib-nano, which cannot print 64-bit integers.
            char sequence[21];
            *std::to_chars(sequence, sequence + sizeof(sequence) - 1, entry.sequence).ptr = '\0';
            output("# error %s up_ms=%" PRIu32 " %s x%" PRIu32 ": %s",
                   sequence, entry.uptime_ms, entry.tag, entry.occurrences, entry.text);
            ++count;
        }
    }
    output("@ok errors %u/%u", count, static_cast<unsigned>(kCapacity));
}
