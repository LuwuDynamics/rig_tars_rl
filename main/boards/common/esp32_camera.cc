#include "sdkconfig.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <esp_heap_caps.h>
#include <esp_log.h>

#include "esp32_camera.h"
#include "board.h"
#include "display.h"
#include "lvgl_display.h"
#include "system_info.h"
#include "jpg/image_to_jpeg.h"
#include "esp_timer.h"

#define TAG "Esp32Camera"

Esp32Camera::Esp32Camera(const camera_config_t &config) {
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed with error 0x%x", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        if (s->id.PID == GC0308_PID) {
            s->set_hmirror(s, 0); // Control camera mirror: 1 for mirror, 0 for normal
        }
        ESP_LOGI(TAG, "Camera initialized: format=%d", config.pixel_format);
    }

    streaming_on_ = true;
}

Esp32Camera::~Esp32Camera() {
    if (streaming_on_) {
        if (current_fb_) {
            esp_camera_fb_return(current_fb_);
            current_fb_ = nullptr;
        }
        if (encode_buf_) {
            heap_caps_free(encode_buf_);
            encode_buf_ = nullptr;
            encode_buf_size_ = 0;
        }
        esp_camera_deinit();
        streaming_on_ = false;
    }
}

void Esp32Camera::SetExplainUrl(const std::string &url, const std::string &token) {
    explain_url_ = url;
    explain_token_ = token;
}

bool Esp32Camera::Capture() {
    if (!streaming_on_) {
        return false;
    }

    // Flush all stale frames to ensure a fresh capture.
    // With fb_count=2 (double buffering), we need 3 iterations:
    //   iter 1: drain buffer A (old), return it
    //   iter 2: drain buffer B (old), return it
    //   iter 3: both buffers now free, camera MUST capture a new frame -> fresh!
    for (int i = 0; i < 3; i++) {
        if (current_fb_) {
            esp_camera_fb_return(current_fb_);
        }
        current_fb_ = esp_camera_fb_get();
        if (!current_fb_) {
            ESP_LOGE(TAG, "Camera capture failed");
            return false;
        }
    }

    // Prepare encode buffer for RGB565 format (with optional byte swapping)
    if (current_fb_->format == PIXFORMAT_RGB565) {
        size_t pixel_count = current_fb_->width * current_fb_->height;
        size_t data_size = pixel_count * 2;

        // Allocate or reallocate encode buffer if needed
        if (encode_buf_size_ < data_size) {
            if (encode_buf_) {
                heap_caps_free(encode_buf_);
            }
            encode_buf_ = (uint8_t *)heap_caps_malloc(data_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (encode_buf_ == nullptr) {
                ESP_LOGE(TAG, "Failed to allocate memory for encode buffer");
                encode_buf_size_ = 0;
                ReleaseFrame();
                return false;
            }
            encode_buf_size_ = data_size;
        }

        // Copy data to encode buffer with optional byte swapping
        uint16_t *src = (uint16_t *)current_fb_->buf;
        uint16_t *dst = (uint16_t *)encode_buf_;
        if (swap_bytes_enabled_) {
            for (size_t i = 0; i < pixel_count; i++) {
                dst[i] = __builtin_bswap16(src[i]);
            }
        } else {
            memcpy(encode_buf_, current_fb_->buf, data_size);
        }

        // Allocate separate buffer for preview display
        uint8_t *preview_data = (uint8_t *)heap_caps_malloc(data_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (preview_data != nullptr) {
            memcpy(preview_data, encode_buf_, data_size);
            auto display = Board::GetInstance().GetDisplay();
            auto lvgl_display = dynamic_cast<LvglDisplay *>(display);
            if (lvgl_display != nullptr) {
                // LvglDisplay 使用专用接口
                lvgl_display->SetPreviewImage(std::make_unique<LvglAllocatedImage>(preview_data, data_size, current_fb_->width, current_fb_->height, current_fb_->width * 2, LV_COLOR_FORMAT_RGB565));
            } else if (display != nullptr) {
                // 其他 Display 使用通用 RGB565 接口
                display->SetPreviewRgb565(preview_data, current_fb_->width, current_fb_->height, current_fb_->width * 2);
                heap_caps_free(preview_data);  // 通用接口会复制数据，释放临时缓冲区
            } else {
                heap_caps_free(preview_data);
            }
        }
    } else if (current_fb_->format == PIXFORMAT_JPEG) {
        // JPEG format preview usually requires decoding, skip preview display for now, just log
        ESP_LOGW(TAG, "JPEG capture success, len=%zu, but not supported for preview", current_fb_->len);
    }

    ESP_LOGI(TAG, "Captured frame: %dx%d, len=%zu, format=%d",
             current_fb_->width, current_fb_->height, current_fb_->len, current_fb_->format);

    return true;
}

bool Esp32Camera::SetHMirror(bool enabled) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        return false;
    }
    s->set_hmirror(s, enabled ? 1 : 0);
    return true;
}

bool Esp32Camera::SetVFlip(bool enabled) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        return false;
    }
    s->set_vflip(s, enabled ? 1 : 0);
    return true;
}

bool Esp32Camera::SetSwapBytes(bool enabled) {
    swap_bytes_enabled_ = enabled;
    return true;
}

std::string Esp32Camera::Explain(const std::string &question) {
    std::unique_ptr<camera_fb_t, decltype(&esp_camera_fb_return)> frame(
        std::exchange(current_fb_, nullptr), &esp_camera_fb_return);
    if (explain_url_.empty()) {
        throw std::runtime_error("Image explain URL or token is not set");
    }

    if (!frame || !frame->buf || frame->len == 0) {
        throw std::runtime_error("No camera frame captured");
    }

    v4l2_pix_fmt_t enc_fmt;
    switch (frame->format) {
        case PIXFORMAT_RGB565:
            enc_fmt = V4L2_PIX_FMT_RGB565;
            break;
        case PIXFORMAT_YUV422:
            enc_fmt = V4L2_PIX_FMT_YUYV;
            break;
        case PIXFORMAT_YUV420:
            enc_fmt = V4L2_PIX_FMT_YUV420;
            break;
        case PIXFORMAT_GRAYSCALE:
            enc_fmt = V4L2_PIX_FMT_GREY;
            break;
        case PIXFORMAT_JPEG:
            enc_fmt = V4L2_PIX_FMT_JPEG;
            break;
        case PIXFORMAT_RGB888:
            enc_fmt = V4L2_PIX_FMT_RGB24;
            break;
        default:
            ESP_LOGE(TAG, "Unsupported pixel format: %d", frame->format);
            throw std::runtime_error("Unsupported camera pixel format");
    }

    const uint16_t width = frame->width;
    const uint16_t height = frame->height;
    const bool use_encode_buffer = frame->format == PIXFORMAT_RGB565 && encode_buf_ != nullptr;
    auto* src = use_encode_buffer ? encode_buf_ : frame->buf;
    const size_t src_len = use_encode_buffer ? size_t{width} * height * sizeof(uint16_t) : frame->len;

    // Complete the encode before opening the upload. The encoder already returns a full JPEG.
    uint8_t* jpeg_data = nullptr;
    size_t jpeg_size = 0;
    const int64_t start_time = esp_timer_get_time();
    const bool encoded = image_to_jpeg(src, src_len, width, height, enc_fmt, 80, &jpeg_data, &jpeg_size);
    std::unique_ptr<uint8_t, decltype(&heap_caps_free)> jpeg(jpeg_data, &heap_caps_free);
    frame.reset();
    if (!encoded || !jpeg || jpeg_size == 0) {
        ESP_LOGE(TAG, "JPEG encoder failed or produced empty output");
        throw std::runtime_error("Failed to encode image to JPEG");
    }
    ESP_LOGI(TAG, "JPEG encoding time: %d ms", static_cast<int>((esp_timer_get_time() - start_time) / 1000));

    auto network = Board::GetInstance().GetNetwork();
    auto http = network->CreateHttp(3);
    std::string boundary = "----ESP32_CAMERA_BOUNDARY";

    http->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());
    http->SetHeader("Client-Id", Board::GetInstance().GetUuid().c_str());
    if (!explain_token_.empty()) {
        http->SetHeader("Authorization", "Bearer " + explain_token_);
    }
    http->SetHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    http->SetHeader("Transfer-Encoding", "chunked");
    if (!http->Open("POST", explain_url_)) {
        ESP_LOGE(TAG, "Failed to connect to explain URL");
        throw std::runtime_error("Failed to connect to explain URL");
    }

    auto write_chunk = [&http](const char* part, const char* data, size_t size) {
        // Wi-Fi HttpClient::Write counts the hex length and both CRLFs, even for the final empty chunk.
        size_t wire_size = size + 5;
        for (size_t length = size; length >= 16; length >>= 4) {
            ++wire_size;
        }
        const int written = http->Write(data, size);
        if (written < 0 || static_cast<size_t>(written) != wire_size) {
            ESP_LOGE(TAG, "Photo upload failed at %s: wrote %d of %zu bytes", part, written, wire_size);
            throw std::runtime_error(std::string("Photo upload failed while sending ") + part);
        }
    };

    {
        std::string question_field;
        question_field += "--" + boundary + "\r\n";
        question_field += "Content-Disposition: form-data; name=\"question\"\r\n";
        question_field += "\r\n";
        question_field += question + "\r\n";
        write_chunk("question", question_field.data(), question_field.size());
    }
    {
        std::string file_header;
        file_header += "--" + boundary + "\r\n";
        file_header += "Content-Disposition: form-data; name=\"file\"; filename=\"camera.jpg\"\r\n";
        file_header += "Content-Type: image/jpeg\r\n";
        file_header += "\r\n";
        write_chunk("file header", file_header.data(), file_header.size());
    }

    constexpr size_t kJpegChunkSize = 2048;
    for (size_t offset = 0; offset < jpeg_size;) {
        const size_t size = std::min(kJpegChunkSize, jpeg_size - offset);
        write_chunk("JPEG data", reinterpret_cast<const char*>(jpeg.get() + offset), size);
        offset += size;
    }
    jpeg.reset();

    {
        const std::string multipart_footer = "\r\n--" + boundary + "--\r\n";
        write_chunk("multipart footer", multipart_footer.data(), multipart_footer.size());
    }
    write_chunk("HTTP terminator", "", 0);

    const int status = http->GetStatusCode();
    const auto request_id = status > 0 ? http->GetResponseHeader("X-Vision-Request-Id") : std::string{};
    if (status != 200) {
        ESP_LOGE(TAG, "Failed to upload photo: HTTP %d, request_id=%s", status, request_id.c_str());
        throw std::runtime_error("Failed to upload photo");
    }

    std::string result = http->ReadAll();
    http->Close();
    ESP_LOGI(TAG, "Explain image: %ux%u, JPEG=%zu bytes, request_id=%s", width, height, jpeg_size, request_id.c_str());
    return result;
}

void Esp32Camera::ReleaseFrame() {
    if (current_fb_) {
        esp_camera_fb_return(current_fb_);
        current_fb_ = nullptr;
    }
}
