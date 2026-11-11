# 技术架构与代码/实验证据映射索引 (Architecture & Experiment Mapping)

本文档整理了关于 **Alicia-D 真机具身策略复现、五大模型横向对比（`总比`）与各模型内部纵向消融（`分比`）** 的核心技术架构，精确映射到本仓库的具体源码文件、函数行号与实验图表，方便技术深挖与快速查阅。

---

## 一、模块一：五大具身策略全流程复现与横向对比（ACT / SmolVLA / $\pi_{0.5}$ / StarVLA / TurboVLA）

### 1. 链路打通：Alicia-D 六轴机械臂底层接入、相机像素级对齐、权重适配与 30Hz 闭环推理
- **硬件底层驱动 (`AliciaDFollower`)**：
  - [`hardware/alicia_d_follower.py`](../hardware/alicia_d_follower.py)：基于 `alicia_d_sdk` 封装 LeRobot 标准 `Robot` 接口，实现 6 轴关节角（弧度 $\leftrightarrow$ 角度自动转换）+ 平行夹爪（`0~1000` 开合度）的 30Hz 状态读取 (`get_observation`) 与平滑下发 (`send_action`)。
  - [`hardware/config_alicia_d_follower.py`](../hardware/config_alicia_d_follower.py)：配置串口、双相机（`top` 全局相机 `/dev/video2` + `wrist` 手腕相机 `/dev/video6`）及单步最大相对变化限制 `max_relative_target`。
- **全局相机在线像素级配准工具 (`align_top_camera.py`)**：
  - [`scripts/align_top_camera.py`](../scripts/align_top_camera.py)：实时加载训练集 `Episode 0 Frame 0` 基准图像，与当前 `/dev/video2` 画面进行 `50/50` 半透明叠加、绿色 Canny 边缘对齐与 `ORB + RANSAC` 像素偏移 `(dx, dy)` 实时测算，确保真机评测前相机视角偏移控制在 `<= 3 px` 以内。
- **多策略统一真机闭环推理入口**：
  - [`scripts/eval_alicia_arms.py`](../scripts/eval_alicia_arms.py)：支持 `ACT`、`SmolVLA`、`pi05` 的一键真机闭环推理、相机自动曝光预热 (`15 frames warmup`)、从 `train_config.json` 自动恢复 `rename_map` 及评测数据集自动录制。
  - [`scripts/eval_alicia_starvla.py`](../scripts/eval_alicia_starvla.py)：支持 `StarVLA (Qwen3.5-2B-VL + Flow-Matching DiT-B)` 的 30Hz 真机闭环推理、`q01/q99` 分位数反归一化与 `RecedingHorizonTrajectoryBlender` Action Chunk 边界连续性平滑。
  - [`scripts/eval_alicia_turbovla.py`](../scripts/eval_alicia_turbovla.py)：支持 `TurboVLA`（ACT 类单步并行解码）策略的 30Hz 真机闭环推理与 `RecedingHorizonTrajectoryBlender` Action Chunk 边界连续性平滑。

### 2. 横向范式对比（`总比`）：五大策略最优配置黄金抓取段定量横评
- **评测曲线图**：[`assets/benchmarks/trajectory_tracking_all_models.png`](../assets/benchmarks/trajectory_tracking_all_models.png)
- **五大策略真机闭环演示动图（30Hz 双视角同步录制）**：
  - `TurboVLA (Warm-Full 12.5k)`：[`assets/demos/rollout_turbovla_smooth.gif`](../assets/demos/rollout_turbovla_smooth.gif)
  - `StarVLA (FixQwen 10k)`：[`assets/demos/rollout_starvla_smooth.gif`](../assets/demos/rollout_starvla_smooth.gif)
  - `π0.5 (Full 20k)`：[`assets/demos/rollout_pi05_rtc.gif`](../assets/demos/rollout_pi05_rtc.gif)
  - `SmolVLA (30k)`：[`assets/demos/rollout_smolvla_ensemble.gif`](../assets/demos/rollout_smolvla_ensemble.gif)
  - `ACT (50k)`：[`assets/demos/rollout_act50_ensemble.gif`](../assets/demos/rollout_act50_ensemble.gif)
- **核心横向定量结论（同一 15 步块执行窗口，Frames 150–300）**：
  - **TurboVLA (`Warm-Full 12.5k`, ACT 类单步并行解码)**：6 轴平均 MAE **0.166°**（RMSE `0.217°`），夹爪 MAE **6.53**，单块推理延迟 **16.8 ms**（RTX 4090，比 $\pi_{0.5}$ / StarVLA 快 **6.8×**），真机闭环表现：**成功**；
  - **$\pi_{0.5}$ (`Full 20k`, PaliGemma-3B + 10-step Flow Matching)**：6 轴平均 MAE **0.382°**（RMSE `0.518°`），夹爪 MAE **19.23**，单块延迟 **112.0 ms**，真机闭环表现：**成功**；
  - **SmolVLA (`30k`, SmolVLM2-500M + 10-step Flow Matching)**：6 轴平均 MAE **0.386°**（RMSE `0.501°`），夹爪 MAE **26.43**，单块延迟 **56.3 ms**，真机闭环表现：**成功**；
  - **StarVLA (`FixQwen 10k`, Frozen Qwen3.5-2B-VL + 10-step DiT-B)**：6 轴平均 MAE **0.639°**（RMSE `0.889°`），夹爪 MAE **16.90**，单块延迟 **115.0 ms**，真机闭环表现：**成功**；
  - **ACT (`50k`, ResNet18 + CVAE)**：6 轴平均 MAE **1.209°**（RMSE `1.494°`），夹爪 MAE **14.12**，单块延迟 **6.1 ms**，真机闭环表现：**成功**。

### 3. 五大模型真机闭环时序平滑横向消融（Temporal Ensembling / Flow-Matching RTC / Action Chunk Boundary Blending）
- **横向对比总图**：[`assets/benchmarks/temporal_smoothing_ablation.png`](../assets/benchmarks/temporal_smoothing_ablation.png)
- **ACT 指数时序加权融合 (`temporal_ensemble_coeff = 0.01`)**：
  - 单步动作变化 $\|\Delta a_t\|_2$ 从 `0.0109 rad` 降至 **`0.0053 rad (-51%)`**，三阶动作抖动 `MeanJerk` 从 `0.5212` 降至 **`0.0408 °/step³ (-92%)`**，最大块边界跳变从 `12.98°` 降至 **`2.45° (-81%)`**，真机表现：**成功**。
- **SmolVLA & $\pi_{0.5}$ 实时分块引导（Flow-Matching RTC Prefix Guidance）**：
  - 将 $\pi_{0.5}$ 的块边界最大单步跳变从 `7.71°` 压降至 **`1.33° (-83%)`**，真机表现：**成功**。
- **StarVLA & TurboVLA Action Chunk 边界连续性混合器 (`RecedingHorizonTrajectoryBlender`)**：
  - **StarVLA (`FixQwen 10k`)**：步间变化 $\|\Delta a_t\|_2$ 从 `0.0137 rad` 降至 **`0.0060 rad (-56%)`**，`MeanJerk` 从 `0.7096` 降至 **`0.0179 °/step³ (-97%)`**，动作方向翻转率从 `54.9%` 降至 **`0.3%`**，最大单步跳变从 `3.15°` 降至 **`0.88° (-72%)`**，真机表现：**成功**；
  - **TurboVLA (`Warm-Full 12.5k`)**：步间变化 $\|\Delta a_t\|_2$ 从 `0.0102 rad` 降至 **`0.0052 rad (-49%)`**，`MeanJerk` 从 `0.4067` 降至 **`0.0151 °/step³ (-96%)`**，动作方向翻转率从 `35.8%` 降至 **`1.2%`**，最大单步跳变从 `5.98°` 降至 **`1.31° (-78%)`**，真机表现：**成功**。

---

## 二、模块二：各策略纵向消融实验（`纵向对比 / 分比`）

### 1. StarVLA 纵向消融：仅冻结视觉塔 (`FreezeViT`) vs. 全冻结 Qwen3.5-2B-VL (`FixQwen`)
- **专页报告与图表**：[`starvla/README.md`](../starvla/README.md)、[`starvla/starvla_3way_freeze_ablation.png`](../starvla/starvla_3way_freeze_ablation.png)、[`starvla/starvla_smooth_vs_no_smooth.png`](../starvla/starvla_smooth_vs_no_smooth.png)、[`starvla/starvla_freeze_vit_smooth_vs_no_smooth.png`](../starvla/starvla_freeze_vit_smooth_vs_no_smooth.png)
- **两路冻结与四组真机闭环对照结论**：
  - **`FreezeViT (10k)`（仅冻结视觉塔）**：`1.93B` 参数可训练，耗时 `74 分钟`（显存 `15.8 GB`），开环 MAE `0.543°`，无平滑 (`eval_starvla_freeze_vit_no_smooth`) 与边界连续性平滑 (`eval_starvla_smooth`) 真机表现均为**成功**（平滑后 `MeanJerk` 从 `0.5927` 降至 `0.0208 °/step³ (-96%)`）；
  - **`FixQwen (10k)`（全冻结 Qwen3.5-2B-VL — 最优配置）**：设置 `freeze_module: "qwen_vl_interface"`，仅训练 `209M` DiT-B 动作头与状态投影层，耗时缩短至 **`44 分钟`（提速 1.7×）**，显存降至 **`9.6 GB`（降低 39%）**，无平滑 (`eval_starvla_fixqwen_no_smooth`) 与边界连续性平滑 (`eval_starvla_fixqwen_smooth`) 真机表现均为**成功**（平滑后 `MeanJerk` 低至 **`0.0179 °/step³ (-97%)`**，动作方向翻转率仅 **`0.3%`**）。

### 2. TurboVLA 纵向五路消融：跨本体选择性热启动与冻结真实 DINOv3
- **专页报告与图表**：[`turbovla/README.md`](../turbovla/README.md)、[`turbovla/turbovla_3way_transfer_ablation.png`](../turbovla/turbovla_3way_transfer_ablation.png)、[`turbovla/turbovla_smooth_vs_no_smooth.png`](../turbovla/turbovla_smooth_vs_no_smooth.png)
- **数据加载与离线 DINOv3 兼容模块**：[`policies/turbovla_alicia/wipes_dataset.py`](../policies/turbovla_alicia/wipes_dataset.py)、[`policies/turbovla_alicia/dinov3_compat.py`](../policies/turbovla_alicia/dinov3_compat.py)
- **五路对照结论**：
  - **Run B (`Scratch Unfrozen`, 20k)** & **Run A (`Sim-ViT Unfrozen Warm-Start`, 20k)**：未冻结视觉主干导致单场景背景记忆或仿真 DINO 特征偏移，真机表现：失败；
  - **Run C (`Frozen Real DINOv3 + Warm-Full`, 8k)**：耗时 `23 分钟`，开环 MAE `0.606°`，真机表现：**成功**；
  - **Run D (`Frozen Real DINOv3 + Warm-Body`, 12.5k)**：仅热启动 Transformer 融合层而随机初始化动作头（`Loss=0.0131`，开环 MAE `0.171°`），真机表现：**成功**；
  - **Run E (`Frozen Real DINOv3 + Warm-Full`, 12.5k — 最优配置)**：同时保留真实世界冻结 `DINOv3` 与 `LIBERO 40k` 预训练 `ACTDecoder` 动作头，Loss 降至 **`0.0110`**，开环 MAE 降至 **`0.166°`**，真机表现：**成功**（且相比 `Warm-Body` 步间 Jerk 降低 **2.2×~2.9×**）。
