# RIG-TARS

面向 RIG-TARS 双足机器人的 ESP32-S3 固件，支持端侧强化学习起身、基于 IMU 的航向保持，以及 MuJoCo 训练和仿真。

[English](README.md) · [固件发布](https://github.com/LuwuDynamics/rig_tars_rl/releases) · [训练与仿真](tars_rl/README.md)

## 实机演示

https://github.com/user-attachments/assets/2629a972-4706-4eba-9549-4180b5689733

## 运动与控制

| 能力 | 实现 |
| --- | --- |
| PPO 强化学习起身 | TARS 检测到倒地后，在 ESP32-S3 上运行训练好的策略尝试起身。默认第一、第三次使用策略，第二次使用关键帧动作。 |
| v9 步态 | 通过 CMA-ES 优化步态参数，支持前进、后退和转向。运动控制器默认以 500 Hz 运行。 |
| IMU 姿态估计 | 使用六轴 QMI8658C，由 Mahony 滤波器融合陀螺仪和加速度计数据，并估计陀螺仪零偏以减少漂移。 |
| 直行航向保持 | 前进或后退时，根据陀螺仪估计相对起步方向的偏航，调整左右髋关节的摆动幅度。直行时默认启用。 |
| 预设动作库 | 固件内置 21 个动作，包括鞠躬、挥手和机器人舞。 |

## 标定与交互

- **一键标定。** 按屏幕动画将机身站直、摆正两条腿，轻点顶部触摸键，同时保存舵机零位和 IMU 标定。
- **语音与遥控。** 支持本地唤醒词、区域语音提示和 BLE 遥控，提供运动、标定和状态查询等 MCP 工具。
- **屏幕与诊断。** 320 × 240 屏幕显示对话状态和机器人遥测信息。相机帧缓存在 PSRAM，Flash 中保留最近 10 条错误记录。
- **重置设置。** 长按顶部触摸键到 10 秒倒计时结束，清除 Wi-Fi 与标定数据。提前松手可取消。重新标定完成后进入配网。

## 固件

请选择与设备区域匹配的版本。烧录包包含应用固件、对应资源、引导程序、分区表和烧录说明。

| 版本 | 语音 | 3.8.1 下载 |
| --- | --- | --- |
| 国内版 | 中文 | [下载国内版](https://github.com/LuwuDynamics/rig_tars_rl/releases/download/v3.8.1/rig-tars-3.8.1-cn.zip) |
| 海外版 | 英文 | [下载海外版](https://github.com/LuwuDynamics/rig_tars_rl/releases/download/v3.8.1/rig-tars-3.8.1-gb.zip) |

### 从源码编译

先激活 ESP-IDF 5.5.3，再运行以下命令。配置文件对应 16 MB Flash 的 ESP32-S3。两个版本共用资源生成目录，请顺序编译。

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

每个构建目录包含 `rig-tars.bin`、`board_assets.bin` 和 ESP-IDF 烧录参数。烧录时使用同一次构建的应用和资源。

## 训练与仿真

MuJoCo 工作区包含机器人模型、舵机拟合数据、行为克隆和 PPO 训练脚本，以及当前固件所用的模型检查点。闭环仿真直接编译设备上的 C 控制器，并加载相同的动作表和起身权重。最终 PPO 检查点可导出为固件使用的 C 权重。[训练、评估与导出命令](tars_rl/README.md)

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`main/boards/tars/`](main/boards/tars) | 板级支持、运动控制、标定、屏幕与语音 |
| [`tars_rl/`](tars_rl) | MuJoCo 模型、训练、检查点与仿真 |
| [`configs/`](configs) | 国内版和海外版构建配置 |
| [`tools/tars_log/`](tools/tars_log) | 串口诊断、遥测采集和主机测试 |
| [`docs/`](docs) | RIG-Omni 平台文档 |

仓库也包含 RIG-Puppy、RIG-Hover 和 RIG-ARM 共用的 RIG-Omni 固件。在 `idf.py menuconfig` 的 **RIG-Omni** 菜单中选择对应板型和固件区域即可编译。

[固件打包工具](tools/package_luwu_firmware.py)生成网页烧录清单，可用 `--board tars --dry-run` 预览打包计划。[语音资源生成工具](tools/generate_tars_voice_pack.py)可生成待试听的替换提示音。设备诊断可用 `idf.py monitor` 或 [TARS 串口工具](tools/tars_log/README.md)。

## 许可证

Apache License, Version 2.0. Copyright © 2024–2026 RIG-Omni Contributors.
