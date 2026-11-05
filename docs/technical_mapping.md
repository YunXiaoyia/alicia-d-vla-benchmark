# 技术架构与代码/实验证据映射索引 (Architecture & Experiment Mapping)

本文档整理了关于 **Alicia-D 真机具身策略复现、跨范式对比与跨本体迁移** 的核心技术架构，精确映射到本仓库的具体源码文件、函数行号与实验图表，方便技术深挖与快速查阅。

---

## 一、模块一：LeRobot 具身策略全流程复现与真机部署（Alicia-D / ACT / SmolVLA / $\pi_{0.5}$ / TurboVLA）

### 1. 链路打通：Alicia-D 六轴机械臂底层接入、相机像素级对齐、LoRA 权重合并与闭环推理
- **硬件底层驱动 (`AliciaDFollower`)**：
  - [`hardware/alicia_d_follower.py`](../hardware/alicia_d_follower.py)：基于 `alicia_d_sdk` 封装 LeRobot 标准 `Robot` 接口，实现 6 轴关节角（弧度 $\leftrightarrow$ 角度自动转换）+ 平行夹爪（`0~1000` 开合度）的 30Hz 状态读取 (`get_observation`) 与平滑下发 (`send_action`)。
  - [`hardware/config_alicia_d_follower.py`](../hardware/config_alicia_d_follower.py)：配置串口、双相机（`top` 全局相机 `/dev/video2` + `wrist` 手腕相机 `/dev/video6`）及单步最大相对变化限制 `max_relative_target`。
- **全局相机在线像素级配准工具 (`align_top_camera.py`)**：
  - [`scripts/align_top_camera.py`](../scripts/align_top_camera.py)：实时加载训练集 `Episode 0 Frame 0` 基准图像，与当前 `/dev/video2` 画面进行 `50/50` 半透明叠加、绿色 Canny 边缘对齐与 `ORB + RANSAC` 像素偏移 `(dx, dy)` 实时测算，确保真机评测前相机视角偏移控制在 `<= 3 px` 以内。
- **$\pi_{0.5}$ (3.3B+) LoRA 微调权重合并与归一化统计打包**：
  - [`policies/lerobot_adapters/merge_pi05_lora.py`](../policies/lerobot_adapters/merge_pi05_lora.py)：将 PEFT LoRA 适配器 `merge_and_unload()` 合入基座模型（共重映射 812 个 `state_dict` 键），并同步打包 `policy_preprocessor_step_3_normalizer_processor.safetensors` 与 `postprocessor`，消除推理期 PEFT 额外开销。
- **多策略统一真机闭环推理入口**：
  - [`scripts/eval_alicia_arms.py`](../scripts/eval_alicia_arms.py)：支持 `ACT`、`SmolVLA`、`pi05` 的一键真机闭环推理、相机自动曝光预热 (`15 frames warmup`)、从 `train_config.json` 自动恢复 `rename_map` 及评测数据集自动录制。
  - [`scripts/eval_alicia_turbovla.py`](../scripts/eval_alicia_turbovla.py)：支持 `TurboVLA` 1-Step MeanFlow 策略的 30Hz 真机闭环推理与 `RecedingHorizonTrajectoryBlender` 五次多项式 $C^2$ 边界锚定平滑。

### 2. 范式对比：统一多模型观测/动作接口与四大模型定量横评
- **跨模型相机键名自适应重映射 (`camera_mapper.py`)**：
  - [`policies/lerobot_adapters/camera_mapper.py`](../policies/lerobot_adapters/camera_mapper.py) & [`scripts/eval_alicia_arms.py` (L188-L195, L286-L298)](../scripts/eval_alicia_arms.py#L188-L195)：自动对比策略期望的视觉特征键名（ACT / TurboVLA 的 `observation.images.top/wrist`、SmolVLA 的 `observation.images.camera1/camera2`、$\pi_{0.5}$ 的 `observation.images.base_0_rgb/left_wrist_0_rgb`），在推理预处理流水线内零拷贝完成键名对齐。
- **四大范式（TurboVLA / $\pi_{0.5}$ / SmolVLA / ACT）黄金抓取段定量横评**：
  - **评测脚本**：[`scripts/eval_and_plot_trajectory.py`](../scripts/eval_and_plot_trajectory.py)
  - **对比曲线图**：[`assets/benchmarks/trajectory_tracking_all_models.png`](../assets/benchmarks/trajectory_tracking_all_models.png) & [`assets/benchmarks/trajectory_tracking_turbovla_vs_pi05.png`](../assets/benchmarks/trajectory_tracking_turbovla_vs_pi05.png)
  - **真机闭环演示动图（30Hz 双视角同步录制）**：
    - `TurboVLA (Frozen-DINOv3 8k)`：[`assets/demos/rollout_turbovla_smooth.gif`](../assets/demos/rollout_turbovla_smooth.gif)
    - `π0.5 (Full 20k)`：[`assets/demos/rollout_pi05_rtc.gif`](../assets/demos/rollout_pi05_rtc.gif)
    - `SmolVLA (30k)`：[`assets/demos/rollout_smolvla_ensemble.gif`](../assets/demos/rollout_smolvla_ensemble.gif)
    - `ACT (50k)`：[`assets/demos/rollout_act50_ensemble.gif`](../assets/demos/rollout_act50_ensemble.gif)
  - **核心定量结论**：
    - **TurboVLA (Frozen-DINOv3 8k, 1-NFE MeanFlow)**：6 轴平均开环 MAE **0.606°**（未冻结版 `0.088°` 但真机视觉过拟合），单块推理延迟 **16.8 ms**（RTX 4090，比 $\pi_{0.5}$ 快 **6.7×**），真机闭环抓取放置成功率 **100%**；
    - **$\pi_{0.5}$ (Full 20k, PaliGemma-3B + 10-step Flow Matching)**：6 轴平均 MAE **1.190°**，夹爪 MAE **44.36**，单块延迟 **112.0 ms**，真机闭环成功率 **100%**；
    - **SmolVLA (30k, SmolVLM2-500M + 10-step Flow Matching)**：6 轴平均 MAE **1.224°**，夹爪 MAE **55.62**，单块延迟 **56.3 ms**，真机闭环成功率 **100%**；
    - **ACT (50k, ResNet18 + CVAE)**：6 轴平均 MAE **1.970°**，夹爪 MAE **90.02**，单块延迟 **6.1 ms**，真机闭环成功率 **100%**。

### 3. 四大模型真机闭环时序平滑消融（Temporal Ensembling / Flow-Matching RTC / Quintic $C^2$ Blender）
- **ACT 指数时序加权融合 (`temporal_ensemble_coeff = 0.01`)**：
  - [`scripts/eval_alicia_arms.py` (L230-L236)](../scripts/eval_alicia_arms.py#L230-L236)：每步重查询 CVAE 并按 $w_i = \exp(-0.01 \cdot i)$ 融合重叠块，使单步动作变化 $\|\Delta a_t\|_2$ 从 `0.0109 rad` 降至 **`0.0051 rad (-53%)`**，最大块边界跳变从 `12.98°` 降至 **`2.47° (-81%)`**。
- **SmolVLA & $\pi_{0.5}$ 实时分块引导（Flow-Matching RTC Prefix Guidance）**：
  - [`scripts/eval_alicia_arms.py` (L237-L275)](../scripts/eval_alicia_arms.py#L237-L275)：在 10 步 Euler ODE 采样器中启用 `RTCConfig` 前缀引导，将 $\pi_{0.5}$ 的块边界最大单步跳变从 `7.71°` 压降至 **`1.33° (-83%)`**。
- **TurboVLA 全视野因果高斯滤波 + 五次多项式 $C^2$ 边界锚定器 (`RecedingHorizonTrajectoryBlender`)**：
  - [`scripts/eval_alicia_turbovla.py` (L92-L193)](../scripts/eval_alicia_turbovla.py#L92-L193)：针对 1-NFE 并行动作查询解码器 (`[Q_0..Q_29]`) 存在的块内锯齿抖动（无平滑 `SignFlipRate = 42.3%`）及块边界 `5.88°` 阶跃跳变（无平滑放置阶段曾打翻绿盘），设计四级因果平滑器：
    1. **块内高斯平滑 ($\sigma=1.8$)**：滤除 30 个并行 Query 间的高频非平滑抖动；
    2. **前半块防回退夹紧 (First-Half Anti-Reversal Clamping)**：防止伺服滞后引起的新块前段反向回撤；
    3. **五次多项式最小加加速度锚定 ($s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5, K=12$)**：保证块切换处的位移、速度、加速度 $C^2$ 连续；
    4. **抓取搬运期夹爪单调防抖锁存**：消除 2-token 本体状态瓶颈导致的夹爪中途微张抖动。
  - **真机实测收益**：`TurboVLA` 真机步间变化 $\|\Delta a_t\|_2$ 从 `0.0111 rad` 降至 **`0.0058 rad (-48%)`**，关节三阶加加速度 `MeanJerk` 从 `0.5103 °/step³` 降至 **`0.0172 °/step³`（降低 97% / 30倍）**，速度符号翻转率从 `42.3%` 降至 **`1.1%`**，最大单步跳变从 `5.88°` 降至 **`1.11°`**。
- **四大模型 8 组真机平滑消融对比图**：
  - [`assets/benchmarks/temporal_smoothing_ablation.png`](../assets/benchmarks/temporal_smoothing_ablation.png)
  - **一键复现实验脚本**：[`scripts/eval_act_smooth.sh`](../scripts/eval_act_smooth.sh)、[`scripts/eval_act_no_smooth.sh`](../scripts/eval_act_no_smooth.sh)、[`scripts/eval_smolvla_smooth.sh`](../scripts/eval_smolvla_smooth.sh)、[`scripts/eval_smolvla_no_smooth.sh`](../scripts/eval_smolvla_no_smooth.sh)、[`scripts/eval_pi05_smooth.sh`](../scripts/eval_pi05_smooth.sh)、[`scripts/eval_pi05_no_smooth.sh`](../scripts/eval_pi05_no_smooth.sh)、[`scripts/eval_turbovla_smooth.sh`](../scripts/eval_turbovla_smooth.sh)、[`scripts/eval_turbovla_no_smooth.sh`](../scripts/eval_turbovla_no_smooth.sh)。

---

## 二、模块二：轻量化 1-Step VLA 跨本体迁移与冻结视觉主干消融（TurboVLA / Alicia-D）

### 1. Alicia-D 真机数据集适配层（67 Episodes / 24,535 Frames）
- **数据集加载器与 8 维状态零填充对齐**：
  - [`policies/turbovla_alicia/wipes_dataset.py`](../policies/turbovla_alicia/wipes_dataset.py)：解析 LeRobot v3.0 格式的双视角 AV1 MP4 与 `.parquet` 轨迹；将 Alicia-D 的 7 维本体状态（6 关节角 + 1 夹爪）自动补零对齐至 LIBERO 预训练模型的 8 维 `state_encoder` 输入（`[J1..J6, Gripper, 0.0]`），实现预训练投影层权重 100% 无损复用。
  - [`policies/turbovla_alicia/wipes_stats.json`](../policies/turbovla_alicia/wipes_stats.json)：24,535 帧真机数据的 Mean/Std/Min/Max 归一化统计量。

### 2. 三路对照消融：从头训练 vs. 未冻结 LIBERO 热启动 vs. 冻结真实 DINOv3 + LIBERO 热启动
- **核心训练源码**：[`policies/turbovla_alicia/train_wipes.py`](../policies/turbovla_alicia/train_wipes.py)
  - **跨步长 `action_queries` 线性插值 (`[12, 256] -> [30, 256]`)**：利用 `F.interpolate(..., size=30, mode='linear', align_corners=True)` 将 LIBERO 40k 预训练模型的 12 步动作查询向量平滑拉伸至真机 30Hz 所需的 30 步（1.0 秒预测视野）。
  - **保留真实世界预训练 DINOv3 (`--keep_real_dinov3`)**：跳过 LIBERO 仿真微调检查点中被 MuJoCo 合成纹理污染的 `backbone.dinov3.*` 权重（参数偏移 `mean |ΔW| = 0.00148`），直接保留 Meta 官方 `LVD-1689M` 真实世界预训练权重。
  - **冻结视觉主干防单场景过拟合 (`--freeze_vision`)**：在 67 条单场景真机数据集上冻结 `DINOv3-ViT-B/16`（`lr_dino = 0.0`），仅微调 6 层 VLA 多模态融合增强器与 1-NFE 动作解码器（`lr_backbone = 5e-5, steps = 8000`）。
- **三路对照完整训练日志与图表**：
  - **Run B（从头训练 20k Steps，未冻结 ViT）**：[`docs/train_wipes_b_scratch.log`](./train_wipes_b_scratch.log)（耗时 `162.8 min`，速度 `1.39 step/s`，最终 Loss `0.0059`，真机泛化失败 `0/5`）
  - **Run A（LIBERO 40k 热启动 20k Steps，未冻结 ViT）**：[`docs/train_wipes_a_warmstart.log`](./train_wipes_a_warmstart.log)（耗时 `162.4 min`，速度 `1.39 step/s`，最终 Loss `0.0053`，开环训练帧 MAE 达 `0.088°` 但因记忆静态光照导致真机偏左 `2.7 cm`、偏高 `3.0 cm` 抓空 `0/5`）
  - **Run C（冻结真实 DINOv3 + LIBERO 40k 热启动 8k Steps）**：[`docs/train_wipes_freeze_vit.log`](./train_wipes_freeze_vit.log) & [`scripts/train_turbovla_wipes_freeze_vit.sh`](../scripts/train_turbovla_wipes_freeze_vit.sh)（耗时仅 **`23.0 min`**，速度 **`5.79 step/s`，提速 7.1 倍**，最终 Loss `0.0340`，开环 MAE **`0.606°`**，**真机闭环抓取放置成功率 100% (`5/5`)**）
  - **三路消融对比图**：[`assets/benchmarks/turbovla_warmstart_vs_scratch.png`](../assets/benchmarks/turbovla_warmstart_vs_scratch.png)

### 3. 跨环境无 `timm` 独立推理兼容层
- **DINOv3 独立兼容层**：
  - [`policies/turbovla_alicia/dinov3_compat.py`](../policies/turbovla_alicia/dinov3_compat.py)：解决真机控制端环境缺少原生 `dinov3_vit` 注册或外网 `torch.hub` 依赖的问题，支持 100% 离线加载 `dinov3-vitb16-pretrain-lvd1689m`，数值误差 `0.0`。
