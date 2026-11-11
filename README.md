# RIG-TARS

ESP32-S3 firmware for the RIG-TARS biped, with an on-device PPO recovery policy, IMU-based heading hold, and a MuJoCo training pipeline.

[简体中文](README_CN.md) · [Firmware releases](https://github.com/LuwuDynamics/rig_tars_rl/releases) · [Training and simulation](tars_rl/README.md)

## Hardware demo

https://github.com/user-attachments/assets/2629a972-4706-4eba-9549-4180b5689733

## Locomotion and control

| Capability | How it works |
| --- | --- |
| PPO fall recovery | TARS detects a fall and runs a trained policy on the ESP32-S3 to try to stand up. The default sequence uses the policy on the first and third attempts, with a keyframe sequence on the second. |
| v9 gait | CMA-ES-optimized gait parameters drive forward and reverse walking and turning. The motion controller runs at 500 Hz by default. |
| IMU attitude estimation | A Mahony filter fuses gyroscope and accelerometer measurements from the six-axis QMI8658C. The controller estimates gyroscope bias to reduce drift. |
| Heading hold | During straight walking, the controller estimates yaw drift from the heading at the start of the move and adjusts the left and right hip strokes. Heading hold is enabled by default for forward and reverse walking. |
| Motion library | The firmware includes 21 preset motions, including bowing, waving, and the robot dance. |

## Setup and interaction

- **One-touch calibration.** Follow the on-screen animation to stand TARS upright and align both legs. Tap the top touch key to save the servo zero positions and IMU alignment together.
- **Voice and remote control.** Local wake-word detection, regional voice prompts, BLE remote control, and MCP tools for motion, calibration, and status.
- **Display and diagnostics.** The 320 × 240 display shows conversation and robot telemetry. Camera frames are buffered in PSRAM, and the firmware keeps the 10 most recent error records in flash.
- **Setup reset.** Hold the top touch key through the 10-second countdown to clear Wi-Fi and calibration data. Releasing early cancels the reset. The device returns to Wi-Fi setup after calibration.

## Firmware

Use the build that matches your device region. Each download includes the application, matching assets, bootloader, partition table, and flashing instructions.

| Edition | Voice | 3.8.1 download |
| --- | --- | --- |
| China | Chinese | [China firmware](https://github.com/LuwuDynamics/rig_tars_rl/releases/download/v3.8.1/rig-tars-3.8.1-cn.zip) |
| Global | English | [Global firmware](https://github.com/LuwuDynamics/rig_tars_rl/releases/download/v3.8.1/rig-tars-3.8.1-gb.zip) |

### Build from source

Activate ESP-IDF 5.5.3 before running these commands. The supplied profiles target ESP32-S3 with 16 MB flash. Build the two regions in sequence because the asset generator uses a shared working directory.

```bash
git clone https://github.com/LuwuDynamics/rig_tars_rl.git
cd rig_tars_rl

mkdir -p build/package-configs
for region in cn gb; do
    profile="$PWD/build/package-configs/sdkconfig.tars-$region"
    cp "configs/tars-$region.sdkconfig" "$profile"
    idf.py -B "build/packages/tars-$region" -DSDKCONFIG="$profile" \
        "-DSDKCONFIG_DEFAULTS=$PWD/sdkconfig.defaults;$PWD/sdkconfig.defaults.esp32s3" \
        -DIDF_TARGET=esp32s3 build || exit 1
done
```

Each build directory contains `rig-tars.bin`, `board_assets.bin`, and the ESP-IDF flash arguments. Flash the application and assets from the same build.

## Training and simulation

The MuJoCo workspace includes the robot model, servo calibration data, behavior cloning and PPO training, and the deployed checkpoints. Its closed-loop evaluation compiles the same C controller and uses the same motion tables and recovery weights as the device. The export tool turns the final PPO checkpoint into the firmware's C weights. [Training and evaluation commands](tars_rl/README.md)

## Repository

| Path | Contents |
| --- | --- |
| [`main/boards/tars/`](main/boards/tars) | Board support, motion control, calibration, display, and audio |
| [`tars_rl/`](tars_rl) | MuJoCo model, training, checkpoints, and simulation |
| [`configs/`](configs) | China and Global build profiles |
| [`tools/tars_log/`](tools/tars_log) | Serial diagnostics, telemetry capture, and host tests |
| [`docs/`](docs) | RIG-Omni platform documentation |

This repository includes the shared RIG-Omni firmware for RIG-Puppy, RIG-Hover, and RIG-ARM. To build for those boards, select the board and firmware region under **RIG-Omni** in `idf.py menuconfig`.

The [firmware packaging tool](tools/package_luwu_firmware.py) generates flashing-site manifests and supports a dry run with `--board tars --dry-run`. The [voice pack generator](tools/generate_tars_voice_pack.py) prepares replacement prompts for review. Use `idf.py monitor` or the [TARS serial tools](tools/tars_log/README.md) for device diagnostics.

## License

Apache License, Version 2.0. Copyright © 2024–2026 RIG-Omni Contributors.
