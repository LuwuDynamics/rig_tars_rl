# Tars 串口日志 / 测试工具

固件里新增了一个小“终端”（走你现在插线的那个 USB 串口），电脑上用 `tars_serial.py` 操作。它能：

- 实时看状态：IMU（加速度/陀螺）、3 个舵机的位置/速度/负载、舵机总线电压（= 电池电压）
- 把 IMU + 舵机回读 + 舵机指令录进内存（IMU 100 Hz，舵机 50~200 Hz），录完自动传到电脑存成文件，并给出分析
- 自动跑舵机**阶跃 / 正弦**测试、经典步态**走/转**测试，同时录数据
- 摆指定姿势、舵机**卸力**（手掰腿看数值）、IMU 轴向标定、学习模式开关、方向自检
- **默认什么都不做**，原有行为不变；只有电脑发命令时才工作

固件部分都在 `main/boards/tars/`（只影响 Tars 板型）：`tars_log.cc`（串口/任务）、`tars_log_shell.c`（命令与会话）、`tars_log_core.c`（环形缓冲/文本格式/测试波形，可移植 C）。

**关于串口**：你这块板的 USB 口接的是 USB‑串口桥（电脑上识别为 `1A86:55D3`，即 WCH CH343，连的是 ESP32‑S3 的 UART0），所以数据走 **115200 波特率**（约 11 KB/s）：一次 `test step`（3 秒，~50 KB）回传约 5 秒，`rec 30`（~370 KB）约 30 秒。要快就加 `--fast`（临时切到 921600，约 8 倍，退出时自动恢复；机器人这边 30 秒无人说话也会自己退回 115200）。原生 USB 口的板子（USB‑Serial/JTAG）固件会自动改走直写 USB 缓冲，更快。

---

## 1. 编译、烧录

已经按 Tars 国内版编译好了，产物在 `build/packages/tars-cn/`（`build/` 被 `.gitignore` 忽略）。改了代码重新编译：

```bash
cd <本仓库根目录>
tools/tars_log/build_tars.sh        # ESP-IDF 5.5.x（IDF_PATH 默认 ~/esp/esp-idf-v5.5.3），配置同 tools/package_luwu_firmware.py 的 Tars 国内版
```

烧录（先 `ls /dev/cu.usbmodem*` 看端口；**先退出 `idf.py monitor` 等占串口的程序**）：

```bash
tools/tars_log/flash_tars.sh /dev/cu.usbmodemXXXX
```

它等价于在 `build/packages/tars-cn` 里执行 `python -m esptool --chip esp32s3 -p <端口> -b 460800 write_flash "@flash_args"`（写 bootloader、分区表、otadata、应用、资源，和 `idf.py flash` 一样），用的是 IDF 环境的 python（有 esptool），不需要编译器在 PATH 里。

> 不要执行 `erase_flash`：舵机零位存在 flash 最后一个扇区（0xFFF000），Wi‑Fi 等设置在 NVS，整片擦除会丢。上面的烧录只写固件分区，零位和设置保留。

烧完板子会重启，串口里多一行：`tars_log: serial log/test tool ready ...`。想看启动日志可以先开 `idf.py -B build/packages/tars-cn -p <端口> monitor`，看完 `Ctrl-]` 退出再用本工具。

## 2. 连接

```bash
python3 tools/tars_log/tars_serial.py check
```

- 需要 `pyserial`（`pip install pyserial`，或直接用 IDF 的 python：`~/.espressif/python_env/idf5.5_py3.11_env/bin/python`）；分析/图表需要 `numpy`（两者都已装在系统 python 里）。
- 端口自动识别（macOS 的 `/dev/cu.usbmodem…`），多个设备时加 `--port /dev/cu.usbmodemXXXX`。
- **同一时间只能有一个程序占着串口**：用本工具前先退出 `idf.py monitor`。
- 工具**不会主动动 DTR/RTS**；但很多开发板的 USB‑串口桥把 DTR/RTS 接到了 EN/IO0，**打开串口本身可能让板子复位**（`idf.py monitor` 也一样）——工具会等它重启完（最多约 12 秒）。需要主动复位时加 `--reset`。

`check` 做一次体检：3 个舵机有没有应答、电池电压、IMU 速率、重力方向、陀螺零偏。**请在机器人静止、直立（或平放）时运行。**

## 3. 第一次试用，建议按这个顺序

### 3.1 体检 + 看实时数据
```bash
python3 tools/tars_log/tars_serial.py check
python3 tools/tars_log/tars_serial.py watch        # Ctrl-C 结束
```
`watch` 每行（学习模式开着时末尾多一个 `st=`，是状态机状态）：`acc`(m/s²) `|a|`(应≈9.8) `gyro`(°/s) `up`(姿态滤波的“上”向量，**没做 IMU 标定前是猜的轴向**) `pos`(舵机原始位置 0~1023) `off`(相对已标定零位的偏移，和 `pose`/`test step` 同一单位) `load`(负载，±1000=满扭矩) `age`(回读多久前的，ms) `V`(电压) `cmd`(最近一次发给舵机的目标)。

### 3.2 确认舵机方向（最重要，一分钟）
手持机器人悬空，依次执行：
```bash
python3 tools/tars_log/tars_serial.py pose 40 0 0 6      # 只动 1 号舵机（+40 计数，约 12°），保持 6 秒
python3 tools/tars_log/tars_serial.py pose 0 40 0 6      # 只动 2 号舵机
python3 tools/tars_log/tars_serial.py pose 0 0 40 6      # 只动 3 号（滑块/升降）
python3 tools/tars_log/tars_serial.py release
```
请告诉我每一步看到的现象（**USB 口朝上、屏幕对着你**拿着它）：
1. `pose 40 0 0`：动的是**你左手边还是右手边**那条腿？脚是向**屏幕一侧（前）**还是向**后**摆？
2. `pose 0 40 0`：同上。
3. `pose 0 0 40`：腿是向上**收**（身体降低）还是向下**伸**（身体抬高）？

当前固件的假设是：1 号 = 机器人左臂，+计数 = 脚向前；2 号 = 右臂，+计数 = 脚向后；3 号 +计数 = 腿向上收。（假设与实际差一次“三个符号同时取反”或左右互换，学习模式里有开关能改。）

另一种看方向的办法——**卸力手掰**：
```bash
python3 tools/tars_log/tars_serial.py torque 0           # 舵机变软（机器人会塌，务必手持）
python3 tools/tars_log/tars_serial.py watch              # 另一个终端开也行；看 off= 随你掰动怎么变
python3 tools/tars_log/tars_serial.py torque 1           # 恢复
```

### 3.3 IMU 轴向：静止采样 + 标定
```bash
python3 tools/tars_log/tars_serial.py test hold 5        # 直立静止 5 秒：噪声、零偏、重力沿哪个传感器轴
python3 tools/tars_log/tars_serial.py cal 0              # 直立静止（学习模式用的 IMU 轴向标定，第 1 步）
python3 tools/tars_log/tars_serial.py cal 1              # 屏幕一侧朝下前倾约 30° 并保持静止（第 2 步），保存进 NVS
```

### 3.4 舵机响应（用来校准仿真里的舵机模型）
```bash
python3 tools/tars_log/tars_serial.py test step 1 40     # 1 号舵机阶跃 0→+40→0→−40→0，每档 0.6 s
python3 tools/tars_log/tars_serial.py test step 2 40
python3 tools/tars_log/tars_serial.py test step 3 60
python3 tools/tars_log/tars_serial.py test sine 1 30 1.5 5   # 1 号舵机 1.5 Hz 正弦 5 个周期（步态是 1.25 Hz）
```
输出里的 `delay`（指令到开始动）、`rise`（10%→90%）、`overshoot`、`peak speed`、`final err`（到位误差）、`peak|load|` 就是我要的数据。机器人**悬空或放平**测，别让腿顶着东西。

### 3.5 记录一次真实走路
```bash
python3 tools/tars_log/tars_serial.py test walk 20 8     # 经典步态前进（幅度 20 计数）8 秒，同时录数据
python3 tools/tars_log/tars_serial.py rec 30             # 录 30 秒，期间你用语音/蓝牙让它做任何动作
```
`rec` 期间可以在另一个终端发 `mark 推了一下` 之类的标记，文件里会带时间戳。

### 3.6 学习模式（引导式准备 + 安全地第一次上地）

**学习模式是什么**：固件里的一套在仿真里训练出来的控制器，**IMU 标定已存时开机约 4 秒后自动打开**（`learn auto`，没存过设置的默认就是 `walk`）。前进/后退/左右转用 CMA-ES（进化策略）在带随机重心/摩擦/推力的校准仿真里搜出来的周期步态 + IMU 俯仰反馈；前进/后退还有陀螺仪**走直**（航向保持）；倾斜超过 60° 持续 0.3 秒判定摔倒，等静止后起立：**默认第 1 次尝试用强化学习（PPO）训练的小神经网络**（真机翻身 0.3~0.6 秒，约 3 秒站稳），没站稳第 2 次改用关键帧起立（`learn getup`）。这些都已在真机上验证过（步态组 v9、走直、网络起立、关键帧起立）。**新机器第一次使用要先做 `setup`（IMU 标定 + 确认舵机方向），标定存好后下一次开机它就会自己站起来。**

开启前必须做完三件事，`setup` 会一步步带你做（约 5 分钟，手在机器人旁边）：

```bash
python3 tools/tars_log/tars_serial.py setup
```

1. **检查**：3 个舵机应答、电压、IMU 三个轴都活着。
2. **IMU 轴向标定**：竖直站立静止（`cal 0`），再向前倾约 30° 静止（`cal 1`），并验证两种姿态下估计出的姿态（竖直时 up≈(0,0,1)，前倾时 up_x≈−0.5）。
3. **舵机方向**：手托机器人（USB 口朝上、屏幕对着你），让 1 号、2 号舵机和滑块各动 4 秒（很小，约 12°），你回答“哪条腿动了、往哪边摆”，我据此写入每个舵机的方向（`map`，断电保存）；再用自检姿势确认一遍。
4. **试运行**：`learn dry`——状态机照常运行，舵机不动；竖直静止应一直是 `STAND`，把机器人仰面放倒应判定 `FALLEN`。

每次启动学习模式（也包括 `watch`）时，姿态滤波器会在机器人**静止 0.3 秒**后启动，并同时估计陀螺零偏（这两台机器是 5~10 °/s，仿真只假设 0.7，已在固件里扣除；`status` 里的 `gbias_cdps` 是估计值，单位 0.01 °/s）。

**第一次真正开启**（机器人放在**软垫**上、手扶着，旁边留出空间；`Ctrl-C`、或 App 里的停止按钮 = 急停）：

```bash
python3 tools/tars_log/tars_serial.py --fast --then "learn on go 25" rec 32   # 先开始录，录制开始后立刻打开学习模式（25 秒后自动关闭），录完自动分析
python3 tools/tars_log/tars_serial.py learn on go 30        # 学习模式站立，30 秒后自动关闭（最长 600 秒）
python3 tools/tars_log/tars_serial.py test walk 20 6        # 学习模式开着时：走 6 秒并录数据，结束给出状态时间线
```

分析里的 “learned behaviour” 会列出状态时间线、摔倒次数、爬起次数和恢复用时。爬起测试：软垫上让机器人**仰面躺下**，用第一条命令（`--then` 保证爬起的第一个动作也在记录里；`--then` 命令被机器人拒绝时录音会提前结束并照常保存）；它应该依次 `FALLEN → GETUP1 → GETUP2A/B/C → STAND`。USB 线要留足余量、手提着靠近插头的一段，爬起时机器人的头会抬起来。

**开机自动站起 / 状态历史 / 调节项 / 步态组**（都在 `learn` 下）：

| 命令 | 作用 |
|---|---|
| `learn auto on` / `off` / `walk go` | 开机约 4 秒后自动打开学习模式（断电保存，下次开机生效）。`on`：站立 + 摔倒后自己爬起，**忽略**遥控/语音的走路指令；`walk go`：同时接受走路指令。**没存过任何设置时默认就是 `walk`**（出厂默认见下面“出厂默认”）；存过 `off` 则保持关。只在开机时 NVS 里已经有 IMU 标定时才会自启：这次开机里刚做完的标定不会让机器人自己动起来。机器人拔掉线后靠电池运行，关机即停。 |
| `learn log` | 学习模式做过什么：状态变化（摔倒、爬起第几次、成功）、开机原因（`boot, last reset: PANIC/BROWNOUT/power-on...`），带时间，最近 160 条，存在 RAM 里，断电丢失。拔线运行后接回线读它。 |
| `errors` | 读取设备 Flash 里最近 10 条报错和异常复位。重复报错合并，含文本参数的错误只保存固定模板。第一次使用需要烧录包含 `tars_err` 的分区表。 |
| `learn gait [0..6] [save]` | 走路用哪一组步态：0 = `v4`（在原仿真里搜的，最早的），1 = `v5`（在用真机数据校准后的仿真里搜的，更稳更安静但更慢），2 = `v5` 里前进换成更快、稍不稳的一版，3 = `v6`（**给装了 10 mm 脚板的机器人**搜的，更快；没装脚板别用），4 = `v7`（同样的校准仿真，目标里加了“车身晃动 / 弹跳”惩罚、步频更低，看起来更平顺）。5 = `v8`（`v7` 的前进 + 真机上试过的平顺后退：前进步态的前后镜像，真机后退晃动比旧后退低约 40%）。6 = `v9`（髋是纯正弦的前进/后退，配合 500 Hz 控制器；真机上前进 20 多厘米、后退 20 cm，比 `v8` 更顺）。走路中不能切换；`save` 保存为开机选择。**没存过时默认是 6（`v9`）。** |
| `learn gait custom <f\|b\|t\|r> <15 个整数>` | 只替换当前步态组里的**一个**步态（前进/后退/左转/右转）：15 个参数按 `q0 a1 b1 a2 b2 z0 c1 d1 c2 d2 f kp_h kd_h kp_z kd_z` 的顺序、**每个乘 1e6 取整**（rad、m、Hz …）。**不保存**：`learn gait N` 或重启就恢复编译进去的那一组。参数超出搜索范围（髋 ±0.3 rad、升降 ±12 mm、步频 0.6~2.0 Hz …）会被拒绝；走路中不能改。`learn gait` 的回复末尾会显示 `custom=fb…`。用 `tars_rl/firmware/custom_gait_cmd.py <f\|b\|t\|r> 参数.json` 生成这串数字，这样试新步态不用重新烧录。 |
| `learn speed [on\|off\|max 70..100]` | 速度参数（默认开）：走路/转向命令的**大小**（舵机计数，就是 MCP 的 `speed`，-23~23）决定步态幅度：≤14 → 70%（约 1.4 cm/s），17 → 约 2.2 cm/s，≥20 → **90%**（约 2.7 cm/s；`max` 可改成 70~100，默认 90，因为 100% 在真机上太容易倒）；符号仍决定方向；变化平滑（0.6/秒）。`off` 则不管命令多大都走满幅度（旧行为）。注意：学习型行为只在 `learn auto walk go`（模式 2）或手动 `learn on go` 时才接受遥控/MCP 的走路命令。 |
| `learn straight [on\|off\|kp kd [trim]]` | 走直（**开机默认开**；`learn straight off` 临时关掉，不保存，重启后又是开）：前进/后退时用**陀螺仪**把航向保持住——把走路开始时的朝向记成 0°，对陀螺积分得到偏航角，两条腿的幅度做一点点不一样（左 `q·(1+ε)`、右 `q·(1−ε)`，ε 最多 ±30%）把它拉回来。`kp`：每弧度航向误差修正的幅度百分比（默认 150，约每偏 1° 修 2.6%；真机上左右幅度差转向的效果只有仿真的 1/3，所以比仿真里的 55 大）；`kd`：每 rad/s 偏航角速度（默认 0，仿真里没有帮助）；`trim`：固定修正，千分比，**正 = 向右转**（前进、后退一样），用来补机器人自己的偏向（真机自然偏航是向右，约 0.3~1.5 °/s）。记住的航向误差限制在 ±15°。带数字的命令会同时打开。校准陀螺零偏用的是站立时自动估计的值，所以偏差靠它抵消的部分会留一点残余（零偏误差 0.1°/s → 每分钟偏 6°）。转弯、站立、起身不受影响。关闭时两条腿幅度完全相同（和以前逐位一致）。量法：`python3 tools/tars_log/yaw_drift.py 站立录音.tlog 走路录音.tlog` 给出走路期间转了多少度（正 = 向左转）。 |
| `learn getup [kf\|alt\|nn] [save]` | 摔倒后用哪种起立（**默认 `alt`**）。`alt` = 第 1、3 次尝试用强化学习（PPO）在仿真里训练出来的小神经网络（28-128-128-3，每 0.1 秒决策一次；输入只有姿态、陀螺、自己上一次的指令和经过的时间；手臂速度限制 8 rad/s），第 2 次用关键帧起立；`nn` = 只用网络；`kf` = 只用关键帧。网络 7 秒内没站稳就放松、等静止，走原来的重试流程。状态名 `GETUP_NN`。**真机验证（2026-10-31）**：翻身 0.3~0.6 秒，多数 2.6~4.9 秒站稳（关键帧 6~10 秒）；网络阶段结束前（站直后约 1~3 秒）它是闭环的，碰它会摆臂，交回 `STAND` 后不再动。`save` 保存为开机选择。只在固件里编进了网络时才有。详见 `tars_rl/REPORT.md` §21。 |
| `learn motion [<名字> [速度%]\|stop]` | **表演动作 / 情绪动作**（原地做，做完自己回到站姿；细节和仿真 / 真机数据见 tars_rl/FUN_MOTIONS.md；**第一次先 50% 速度**）：不带参数列出名字：动作 `nod`（点头）、`bow`（鞠躬 55°）、`headbang`（甩头）——**这三个已在真机上以 50% 和 100% 速度各做过一次**（俯仰角和指令吻合，没摔；打喷嚏全速时把整机带翻了，已去掉）；情绪 `happy` `sad` `angry` `scared` `surprised` `shy` `curious` `sleepy` `proud` `laugh` `loving` `impatient`（开心、难过、生气、害怕、惊讶、害羞、好奇、犯困、得意、大笑、撒娇、不耐烦，5~12 秒；真机 50% 和 100% 都做过）；手势 / 舞蹈 `wave`（招手）`cheer`（欢呼）`windmill`（风车）`robotdance`（机器人舞）`swim`（游泳）`twist`（扭一扭）——**会先把两条臂收起来（升降 +14 mm，站在躯干底座上，这个姿势真机上 8 秒保持过：俯仰 ±4° 内），再自由挥动；这 6 个动作本身真机还没做过，第一次先 50%**；`learn motion bow 50` 以 50% 速度做（30..100，默认 100）；`learn motion stop` 立刻停下（髋先回正、升降最后回位）。**要学习模式开着**；机器人必须站着不动（正在走路的话会等它停下，最多 3 秒）；摔倒 / 爬起中 / 正在做别的动作时被拒绝。做的时候两条臂像腿一样撑在地上，躯干挂在髋轴上前后点头，所以要**平整、不打滑的地面**，周围留 20 cm。中途收到走路命令、`stop`、`estop` 或倾角超过动作自己的上限（点头 40°、鞠躬 75°、甩头 65°，情绪 40~55°），或者腿被带着倾斜（本体俯仰和髋角的差超过 30°，整机在翘）都会结束它（后者按摔倒处理，自动爬起）。MCP：`self.tars.motion(name, speed)`，语音说“鞠个躬”“点头”“开心一下”“难过一个”“装害怕”“打个哈欠”（名字也认中文）。日志里是状态 `MOTION`，`learn log` 有 `motion bow requested` / `motion ended (...)`。 |
| `learn tune [kp% kd% lpf_hz amp% hz]` | 走路反馈调节（**不保存**，开机恢复默认 `100 100 0 100 500`）：俯仰反馈、俯仰角速度（陀螺）反馈、陀螺低通、步态幅度、控制频率 50/100/250/500 Hz（500 = 经典步态的指令频率，舵机指令每次只变约 0.6 个计数，IMU/姿态滤波仍是 100 Hz）。**注意**：把陀螺项关到 0 会走几步就摔（它是阻尼）。 |
| `hello` | 现在会显示 `rst=` 上次重启的原因（power-on / BROWNOUT 欠压 / PANIC 崩溃 / 看门狗 …），排查突然重启时看它。 |

> 注意：学习模式开着时，把机器人在手里倾斜超过 60° 会触发“爬起”动作（腿会大幅摆动），测试倾斜请用 `learn dry`。爬起动作最大约 458°/s，注意相机排线和手指。

---

### 3.7 出厂默认（新固件，没存过任何设置时）

| 项目 | 默认 |
|---|---|
| 开机 | IMU 标定已存的话，约 4 秒后自动打开学习模式并接受走路指令（`learn auto walk go`） |
| 摔倒起立 | 学习模式里的摔倒检测；起立 = 强化学习网络第 1 次尝试（真机翻身 0.3~0.6 秒，约 3 秒站稳），没站稳第 2 次改用关键帧（14 个关键帧 + 两段式收尾，真机连续 6 次 2 次内站起）（`learn getup alt`） |
| 前进 / 后退 / 左转 / 右转 | 步态组 6 = `v9`（右转 = 左转的镜像）；走路/转向命令的大小决定幅度：≤14 计数 70%，≥20 计数 90% |
| 走直 | 开（陀螺航向保持，`learn straight off` 可临时关） |
| 控制器频率 | 500 Hz（IMU / 姿态滤波仍是 100 Hz） |
| 其它 | `learn gait N save`、`learn auto off`、`learn getup kf save` 等存过的设置仍然优先 |

**新机器首次使用**：先做 `setup`（IMU 标定 + 确认舵机方向）。标定存好之后下一次开机，机器人就会自己站立并响应走路命令，所以舵机方向没确认前不要重启。

## 4. 命令一览

`tars_serial.py` 的第一个词如果不是 `ports / check / term / analyze`，就原样发给机器人当命令。`term` 是交互终端（直接敲命令，录音自动保存+分析）。

| 命令 | 作用 |
|---|---|
| `check` | 体检（电脑端命令，内部用 `scan` + `test hold 4`） |
| `setup` | 学习模式的引导式准备（电脑端命令）：检查 → IMU 标定 → 舵机方向 → 试运行；可只做某几步：`setup imu`、`setup map`、`setup dry` |
| `scan` | 读一次 3 个舵机 + 总线电压，没应答的会写 `NO REPLY` |
| `status` / `hello` / `help` | 状态 / 版本信息 / 命令表 |
| `watch [hz]` | 实时状态行（默认 5 Hz），`--seconds N` 自动结束 |
| `rec <秒>` | 录 N 秒（0 = 直到 `stop`），结束自动传回并分析 |
| `live` | 实时原始事件流（USB 口才跟得上），存成文件 |
| `stop` | 结束当前录音/监视等任务，并释放被占用的舵机 |
| `estop` | **急停**（电脑端 Ctrl-C 发的就是它）：停走路、关学习模式（腿回零位）、结束任务 |
| `dump` / `clear` | 重新传上一次录音 / 清空 |
| `mark <文字>` | 在录音里打标记 |
| `test hold <秒>` | 静止采样：噪声、零偏、电压 |
| `test step <id> <幅度> [每档ms]` | 舵机阶跃：0→+A→0→−A→0（默认每档 600 ms） |
| `test sine <id> <幅度> <Hz> [周期数]` | 舵机正弦跟踪，给出增益和相位滞后 |
| `test walk <vx> <秒>` / `test turn <vyaw> <秒>` | 经典步态走/转（vx −35..35，vyaw −25..25）并录数据 |
| `pose <c1> <c2> <c3> [秒]` | 三个舵机摆到偏移姿势（计数，相对零位），默认保持 10 秒 |
| `release` | 释放舵机，经典步态恢复 |
| `torque 0\|1` | 0 = 舵机卸力（能手掰），1 = 恢复（腿会弹回零位） |
| `cal 0\|1` | IMU 轴向标定两步 |
| `learn dry` / `learn on go` / `learn off` | 学习模式：只计算 / 真正驱动（需确认词 `go`）/ 关闭 |
| `map <flip> <swap>` | 舵机方向假设（写入 NVS），同 MCP 的 `learned_mapping` |
| `selftest 0..3` | 1 = 双脚向前摆 20°，2 = 腿向上收 12 mm，3 = 零位，0 = 释放（姿势保持 30 秒） |
| `imu [regs\|raw [n]\|reinit\|fs <0..3>]` | QMI8658 IMU 寄存器诊断：`regs` 读配置/状态/原始数据，`raw` 连续读原始值，`reinit` 软复位后恢复出厂同款配置，`fs` 临时改加速度计量程（重启或 `reinit` 恢复） |
| `link usb\|console` | 回传通道：`usb` = 直接写串口发送缓冲（默认，自动选）；`console` = 走普通控制台输出（备用，更慢） |
| `baud <速率>` | UART 速率（115200 / 230400 / 460800 / 921600 / 1500000 / 2000000）；电脑端用 `--fast`，不用手动敲。切换后 8 秒内没收到 `baud ok` 会自动退回 115200 |
| `reboot` | 重启 |

单位：舵机**计数** = 舵机转角，1024 计数 = 300°（1 计数 = 0.293°）。`id`：1、2 = 两个髋（摆腿），3 = 升降滑块。

**限幅**（固件里强制）：髋 ±120 计数（±35°）、滑块 ±110 计数（≈±14 mm，经典步态的最大值）、姿势最长保持 120 秒、卸力最长 600 秒（到时自动恢复）。`pose`/`test` 在舵机零位标定模式下一律拒绝。

## 5. 录音文件和分析

默认存到 `tools/tars_log/tars_logs/时间_类型.tlog`（纯文本，已被 `.gitignore` 忽略；用 `--out-dir` 改位置）。里面是机器人发来的原始行：

```
@I,t_us,ax,ay,az,gx,gy,gz,ux,uy,uz,flags,state,vxy*HH    IMU（原始 LSB：加速度 9.8/16384 m/s²，陀螺 1/32 °/s，up ×10000）
@S,t_us,id,pos,spd,load,lat_us*HH                          舵机回读（原始寄存器；lat = 发出读请求到收到应答的微秒数）
@C,t_us,src,p1,p2,p3,speed*HH                              发给舵机的目标（原始寄存器；src 1 = 本工具的测试波形）
@V,t_us,厘伏*HH    @M,t_us,文字*HH                         电压 / 标记
```
每行末尾 `*HH` 是校验，坏行会被丢弃并计数。`flags`：bit0 姿态滤波已启动，bit1 已做 IMU 标定，bit2 学习模式开，bit3 dry-run，bit4 IMU 没初始化。`vxy` 是当时的前进/转向命令（两个 int8）。

```bash
python3 tools/tars_log/tars_serial.py analyze tools/tars_log/tars_logs/xxx.tlog                  # 文字分析
python3 tools/tars_log/tars_serial.py analyze tools/tars_log/tars_logs/xxx.tlog --html r.html    # 带曲线的网页报告（纯 SVG，不依赖库）
python3 tools/tars_log/tars_serial.py analyze tools/tars_log/tars_logs/xxx.tlog --csv flat.csv   # 按 IMU 时间轴对齐的 CSV
```
舵机速度/负载寄存器里的方向位（速度 bit15、负载 bit10）按飞特 SCS 协议解码；原始值都在文件里，以后改解码不用重录。

## 6. 安全

- `test step / sine`、`pose`、`selftest` 会**让舵机动**：手持或让机器人仰躺，先用小幅度（30~40）。
- `test walk / turn` 会**真的走**：放在软垫上，手放在旁边；`Ctrl-C` 或 `stop` 立即停。
- `torque 0` 会让机器人**塌下来**：务必手持。
- `learn on go` 会让机器人按学出来的步态/起立自己动；新机器先 `learn dry`，且需要先完成 `cal 0` / `cal 1` 并确认舵机方向（`setup`）。
- 舵机有保护：>80% 堵转持续 2 s、电压 <4.5 V 会保护。别让腿顶着障碍物保持 `pose`。
- 学习模式/日志工具都只在你发命令后才起作用；断电或 `reboot` 后恢复原样。

### 5.1 其它小工具

| 工具 | 用途 |
|---|---|
| `yaw_drift.py 站立录音.tlog 走路录音.tlog…` | 走路期间机器人转了多少度（陀螺对竖直轴积分；站立录音提供零偏）。正 = 向左转（用 TURN_L / TURN_R 的录音核对过）。用来测"走不直"和 `learn straight` 的效果 |
| `console_capture.py [--out 文件]` | 被动记录机器人从串口打印的所有文字（不碰 DTR/RTS，不会复位）并加上电脑时间，用来抓**崩溃时的寄存器/回溯**（`rst=PANIC`）——别的工具在机器人闲着时看不到。它占着串口，用 `tars_serial.py` 之前要先停掉（Ctrl-C） |

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `does not answer 'hello'` | 固件没烧新的 / 串口被 `idf.py monitor` 占用 / 板子卡在下载模式或重启循环（按 RESET 再试，或加 `--reset`）。工具会依次试 115200 和 921600/460800/230400，再自动盲发 `link console` 走备用通道；也可以一开始就加 `--console` |
| `Cannot open ...` | 端口被别的程序占着；换 `--port` |
| 屏幕/声音在传大文件时卡顿 | 回传期间日志任务占一点 CPU；传完就好，或用 `--fast` 缩短时间 |
| `servo N: NO REPLY` | 该舵机线、ID 或供电有问题（老固件每 100 ms 才读一个舵机，不容易发现） |
| `IMU not initialised` | I2C 没通；重新上电 |
| `[FAIL] accelerometer z axis is stuck at the full-scale code` | 该加速度计通道坏了（输出固定 0x8000，换量程、软复位都不变）。`imu regs` / `imu fs 1` / `imu reinit` 可复查。`cal` 会拒绝执行（否则会存下一个垃圾的轴向矩阵） |
| `@err ...: busy` | 上一个任务还没结束，先 `stop` |
| 录音传一半断了 | 机器人会等你最多 8 秒，之后放弃；发 `dump` 重传（数据还在 RAM 里） |
| Windows | 用 `--port COM5`，其余相同 |

## 8. 测试（改代码时用）

固件里的命令外壳和核心是可移植 C，不接硬件就能测：

```bash
make -C tools/tars_log/test test
```

另有陀螺零偏估计器的单元测试（`test_gyro_bias`）、学习型监督器的测试（`test_ctrl`，159 项：摔倒/起立/FAILED、走路速度命令、走直闭环、网络起立状态、表演动作的 `MOTION` 状态——按表播放、速度、中止顺序、倾角上限、请求等待 / 超时 / 被拒绝）和起立网络的 C 前向 vs PyTorch 对照（`test_getup_nn`，需要导出的权重和测试向量）。会编译 `tars_log_sim`（真实的 `tars_log_shell.c/core.c` + 一个模拟机器人：舵机动力学、IMU、回读延迟），跑 C 单元测试（环形缓冲多线程压力、格式化、解析、波形）和 51 个端到端测试（含引导式准备的 7 种情形，用脚本化的回答驱动）（用真正的 `tars_serial.py` 经伪终端连上去：体检/阶跃/正弦/走路、busy/限幅/错误、卸力、姿势保持、主机中途停读、随机乱发命令、`--then`……）。**没有测的**是 ESP32 专有的一层（`tars_log.cc`：串口发送/接收 FIFO、波特率切换、任务调度、硬件钩子）——它只能在真机上验证，所以第一次试用时如果有异常，请把终端输出和 `watch` 的几行发给我。

## 9. 做完后请发我什么

1. `check` 的整段输出（含 `[warn]`/`[FAIL]` 行）。
2. `pose` 三个观察（3.2）：每个动的是哪边的腿、方向。
3. `cal 0` / `cal 1` 的输出（含 `# [rl] ...` 几行）。
4. `tools/tars_log/tars_logs/` 里的这些文件（纯文本，直接贴路径或发文件）：`test hold 5`、`test step 1 40`、`test step 2 40`、`test step 3 60`、`test sine 1 30 1.5 5`、`test walk 20 8`。
5. 走路那次：实际走了多远、花了多久、有没有摔、地面材质、机器人是否直立开始。
6. 异常的话：终端里完整的报错/输出，以及发生异常时的操作。

