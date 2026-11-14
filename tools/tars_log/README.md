# TARS 编译、烧录与串口诊断

以下命令在仓库根目录运行。

## 编译和烧录

使用 ESP-IDF 5.5.3。国内版快捷脚本将固件编译到 `build/packages/tars-cn/`。

```bash
tools/tars_log/build_tars.sh
tools/tars_log/flash_tars.sh /dev/cu.usbmodemXXXX
```

将示例端口换成机器人实际的端口。烧录脚本按 `flash_args` 写入本次构建的应用、资源和引导分区，保留舵机零位和 Wi-Fi 设置。

海外版按[仓库的 ESP-IDF 编译命令](../../README_CN.md#从源码编译)完成构建后，在已激活 ESP-IDF 的终端执行以下命令。

```bash
(
    cd build/packages/tars-gb &&
    python -m esptool --chip esp32s3 -p /dev/cu.usbmodemXXXX -b 460800 \
        --before default_reset --after hard_reset write_flash @flash_args
)
```

## 查舵机应答

给机器人上电并接上 USB 数据线。如果已打开 `idf.py monitor`，先用 `Ctrl-]` 退出，让出串口。

```bash
python3 -m pip install pyserial
python3 tools/tars_log/tars_serial.py scan
```

`scan` 会查询 1、2、3 号舵机和总线电压。出现 `servo N: NO REPLY` 时，检查对应舵机的接线、供电和 ID。可用 `ports` 列出串口，用 `--port` 指定设备。

## 查看状态和错误

```bash
python3 tools/tars_log/tars_serial.py status
python3 tools/tars_log/tars_serial.py watch --seconds 5
python3 tools/tars_log/tars_serial.py errors
```

`status` 查看固件当前状态，`watch` 实时显示姿态、舵机反馈和电压，`errors` 读取设备保存的近期错误。用 `Ctrl-C` 中断正在执行的串口命令时，工具会向机器人发送急停指令。启动日志可用 `idf.py monitor` 查看。
