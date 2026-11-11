# TARS 强化学习与 MuJoCo 仿真

TARS 3.8.1 使用 CMA-ES 优化的 v9 步态。起身策略是一个 `28 → 128 → 128 → 3` 的 PPO 网络。固件默认在第一次和第三次起身尝试中使用网络，第二次使用关键帧。MuJoCo 闭环仿真直接编译 [TARS 板级目录](../main/boards/tars) 中的 C 控制器，并加载固件使用的起身权重和 21 个动作表。

## 安装

在 macOS 或 Linux 上使用 Python 3.11 和 `cc`。以下命令均在 `tars_rl/` 下运行。渲染视频时还需安装 FFmpeg。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 仿真验证

```bash
# 用固件默认策略从仰躺和俯卧起身
python rl/eval_getup_c.py --modes alt --plants cal --kinds back,front --n 4 --procs 2

# 在相同仿真环境评估 PyTorch 检查点
python rl/eval_getup.py --policy checkpoints/getup_ppo_c3/final_it500.pt \
    --no-keyframes --plants cal --kinds back,front --n 4 --procs 2

# 检查固件动作及 v9 前进、后退、转向
python rl/fun_motions_c.py check --n 2
python calib/eval_cal.py --n 2 --procs 2 --only walk --skip-original
```

`model/` 保存 MJCF 模型和网格。`calib/servo_fit_v1.json` 提供舵机拟合参数，`data/` 保存关键帧和步态参数。`runs/` 用于保存本地训练和评估结果。

## 训练与导出

用固件关键帧在随机化仿真中采集起身示范，随后训练行为克隆模型。

```bash
python rl/bc_getup.py collect --n 800 --out runs/getup_bc/demos.npz
python rl/bc_getup.py train --demos runs/getup_bc/demos.npz --out runs/getup_bc/bc.pt
```

仓库保留了行为克隆检查点 `bc2.pt` 及后续三阶段 PPO 的检查点 `it160_saturated_lift.pt`、`final_it600.pt` 和 `final_it500.pt`。每个检查点都保存了当时的训练参数。下面从第二阶段权重开始运行最终阶段。

```bash
python rl/ppo_getup.py --init checkpoints/getup_ppo_c2/final_it600.pt \
    --out runs/getup_ppo_c3 --iters 500 --seed 0 --lr 0.00015 --lr_max 0.0003 \
    --reset_std 1 --log_std0 -2.0 --print_every 10 \
    --env_kw '{"reward_mode":"v2","continue_after_success":false,"act_pen":0.3,"obs_q_mode":"cmd","imu_clip":true,"vel_cap":7.5,"vel_pen":0.03,"p_weak":0.25}'
```

用随仓库提供的最终检查点复现固件 C 权重。导出还会生成供主机测试使用的观测和动作向量。

```bash
python rl/export_getup_c.py checkpoints/getup_ppo_c3/final_it500.pt runs/export
cmp runs/export/tars_rl_getup_nn_weights.h ../main/boards/tars/tars_rl_getup_nn_weights.h
python firmware/gen_motions.py --out ../main/boards/tars --check
```

重新搜索步态可从已有参数出发。生成的步态参数可用 `firmware/gen_params_set.py` 导出为 C 头文件。

```bash
python rl/walk_cma_cal.py fwd --gens 60 --pop 28 --procs 8 \
    --x0 data/walk_cma/fwd_v4.npy --sigma 0.15 --tag _c1
```
