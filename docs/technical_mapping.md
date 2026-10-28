# 技术架构与代码/实验证据映射索引 (Architecture & Experiment Mapping)

本文档整理了关于 **Alicia-D 真机具身策略复现、跨范式对比与跨本体迁移** 的核心技术架构，精确映射到本仓库的具体源码文件、函数行号与实验图表，方便技术深挖与快速查阅。

---

## 一、模块一：LeRobot 具身策略全流程复现与真机部署（Alicia-D / ACT / SmolVLA / $\pi_{0.5}$）

### 1. 链路打通：Alicia-D 六轴机械臂底层接入、LoRA 权重合并与闭环推理
- **硬件底层驱动 (`AliciaDFollower`)**：
  - [`hardware/alicia_d_follower.py`](../hardware/alicia_d_follower.py)：基于 `alicia_d_sdk` 封装 LeRobot 标准 `Robot` 接口，实现 6 轴关节角（弧度 $\leftrightarrow$ 角度自动转换）+ 平行夹爪（`0~1000` 开合度）的 30Hz 状态读取 (`get_observation`) 与平滑下发 (`send_action`)。
  - [`hardware/config_alicia_d_follower.py`](../hardware/config_alicia_d_follower.py)：配置串口、双相机（`top` 全局相机 + `wrist` RealSense D405 手腕相机）及单步最大相对变化限制 `max_relative_target`。
- **$\pi_{0.5}$ (4.14B) LoRA 微调权重合并与归一化统计打包**：
  - [`policies/lerobot_adapters/merge_pi05_lora.py`](../policies/lerobot_adapters/merge_pi05_lora.py)：将 PEFT LoRA 适配器 `merge_and_unload()` 合入 4.14B 基座（共重映射 812 个 state_dict 键），并同步打包 `policy_preprocessor_step_3_normalizer_processor.safetensors` 与 `postprocessor`，消除推理期 PEFT 额外开销（合并日志见 [`docs/train_wipes_a_warmstart.log`](./train_wipes_a_warmstart.log) 同级记录）。
- **多策略统一真机闭环推理入口**：
  - [`scripts/eval_alicia_arms.py`](../scripts/eval_alicia_arms.py)：支持 `ACT`、`SmolVLA`、`pi0`、`pi05` 四类策略的一键真机闭环推理、预热 (`warmup`)、安全回零与评测数据集自动录制。

### 2. 范式对比：统一多模型观测/动作接口与四大模型定量横评
- **跨模型相机键名自适应重映射 (`camera_key_mapping`)**：
  - [`scripts/eval_alicia_arms.py` (L239-L260)](../scripts/eval_alicia_arms.py#L239-L260)：自动对比策略期望的视觉特征键名（如 ACT 的 `observation.images.top/wrist`、SmolVLA 的 `observation.images.camera1/camera2`、$\pi_{0.5}$ 的 `observation.images.base_0_rgb/left_wrist_0_rgb`），在推理环内零拷贝完成键名对齐。
- **四大范式（TurboVLA / SmolVLA / $\pi_{0.5}$ / ACT）黄金抓取段定量横评**：
  - **评测脚本**：[`scripts/eval_and_plot_trajectory.py`](../scripts/eval_and_plot_trajectory.py)
  - **对比曲线图**：[`assets/benchmarks/trajectory_tracking_all_models.png`](../assets/benchmarks/trajectory_tracking_all_models.png)
  - **核心结论**：
    - **TurboVLA (0.2B, DINOv3 + BERT + Bi-CrossAttn + ACT Head)**：6 轴平均 MAE **0.088°**，夹爪 MAE **3.63**，单块延迟 **104.8 ms**；
    - **SmolVLA (0.45B, SmolVLM2 + Flow Matching)**：6 轴平均 MAE **0.227°**，夹爪 MAE **9.74**，单块延迟 **19.6 ms**（最快）；
    - **$\pi_{0.5}$ (4.14B, PaliGemma + Action Expert Flow Matching)**：6 轴平均 MAE **0.389°**，夹爪 MAE **17.45**，单块延迟 **551.3 ms**；
    - **ACT (0.08B, ResNet18 + CVAE Transformer)**：6 轴平均 MAE **1.209°**，夹爪 MAE **14.12**，单块延迟 **22.6 ms**。

### 3. 平滑与调优：`ACTTemporalEnsembler` 时序加权平滑与 $\pi_{0.5}$ RTC 前缀引导
- **SmolVLA 外部接入 `ACTTemporalEnsembler`**：
  - [`scripts/eval_alicia_arms.py` (L190-L209, L317-L328)](../scripts/eval_alicia_arms.py#L190-L209)：针对 SmolVLA 原生仅支持 chunk 队列弹出导致的 chunk 边界跳变，在推理环外挂载 `ACTTemporalEnsembler(temporal_ensemble_coeff, chunk_size)`，强制每步调用 `_get_action_chunk()` 并按指数权重 $w_i = \exp(-m \cdot i)$ 融合历史重叠预测。
- **$\pi_{0.5}$ 实时分块引导（RTC Prefix Guidance）**：
  - [`scripts/eval_alicia_arms.py` (L330-L370)](../scripts/eval_alicia_arms.py#L330-L370)：在每跑完 `n_action_steps=15` 步重规划时，截取上一轮未执行完的剩余 15 步作为 `prev_chunk_left_over` 传入 Flow Matching ODE 采样器，消除 551ms 推理延迟带来的块间断层。
- **真机平滑消融曲线图**：
  - [`assets/benchmarks/temporal_smoothing_ablation.png`](../assets/benchmarks/temporal_smoothing_ablation.png)

---

## 二、模块二：轻量化 VLA 策略复现、仿真泛化评测与真机迁移（TurboVLA / Alicia-D）

### 1. Alicia-D 真机数据集适配层（71 Episodes / 24,535 Frames）
- **数据集加载器与 8 维状态对齐**：
  - [`policies/turbovla_alicia/wipes_dataset.py`](../policies/turbovla_alicia/wipes_dataset.py)：解析 LeRobot v3.0 格式的双视角 MP4 与 `.parquet` 轨迹；将 Alicia-D 的 7 维本体状态（6 关节角 + 1 夹爪）自动补零对齐至 LIBERO 预训练模型的 8 维 `state_encoder` 输入（`[J1..J6, Gripper, 0.0]`），实现 `state_encoder` 权重 100% 无损复用。
  - [`policies/turbovla_alicia/wipes_stats.json`](../policies/turbovla_alicia/wipes_stats.json)：24,535 帧真机数据的 Mean/Std/Min/Max 归一化统计量。

### 2. 解码器 `action_queries` 线性插值对齐（`[12, 256] -> [30, 256]`）与热启动消融
- **跨步长线性插值核心实现**：
  - [`policies/turbovla_alicia/train_wipes.py` (L128-L148)](../policies/turbovla_alicia/train_wipes.py#L128-L148)：利用 `F.interpolate(..., size=30, mode='linear', align_corners=True)` 将 LIBERO 40k 预训练模型的 12 步动作查询向量平滑拉伸至真机 30Hz 所需的 30 步（1.0 秒预测视野），达成 `Missing=0, Unexpected=0` 全参数热启动。
- **从头训练 vs LIBERO 40k 热启动 20k Steps 对照消融**：
  - **自动化消融流水线脚本**：[`scripts/train_turbovla_wipes_b_then_a.sh`](../scripts/train_turbovla_wipes_b_then_a.sh)
  - **完整训练日志**：[`docs/train_wipes_a_warmstart.log`](./train_wipes_a_warmstart.log) vs [`docs/train_wipes_b_scratch.log`](./train_wipes_b_scratch.log)
  - **消融对比图**：[`assets/benchmarks/turbovla_warmstart_vs_scratch.png`](../assets/benchmarks/turbovla_warmstart_vs_scratch.png)（收敛 Loss 从 `0.0059` 降至 `0.0053`，降低 **10.2%**；最大梯度范数从 `1.67` 降至 `0.73`）。

### 3. 跨环境无 `timm` 独立推理兼容与真机闭环部署
- **DINOv3 独立兼容层**：
  - [`policies/turbovla_alicia/dinov3_compat.py`](../policies/turbovla_alicia/dinov3_compat.py)：解决真机控制端 `lerobot` 环境（`transformers 4.53.3`）缺少 `dinov3_vit` 原生注册与 `timm` 依赖的问题，实现 RoPE 旋转位置编码与 `layer. -> model.layer.` 键名自动映射，数值误差 `0.0`。
- **TurboVLA 真机闭环推理脚本**：
  - [`scripts/eval_alicia_turbovla.py`](../scripts/eval_alicia_turbovla.py)
