# StarVLA (`Qwen3.5-2B-VL` + Flow-Matching `DiT-B` / `QwenGR00T`) — Alicia-D Real-Robot Deep Dive

[⬅️ 返回总览对比 (Back to Main Benchmark README)](../README.md)

---

## 1. 模型架构与跨本体适配规格 (Architecture & Cross-Embodiment Spec)

| 属性 (Property) | 规格配置 (Specification) |
| :--- | :--- |
| **生成范式 (Generative Paradigm)** | **Flow-Matching Diffusion Transformer (`DiT-B`, 10 Euler ODE Steps)** + Autoregressive VLM Hidden-State Conditioning (`QwenGR00T`) |
| **多模态主干 (VLM Backbone)** | **Frozen `Qwen3.5-2B-VL`**（`337M` 视觉编码器 `visual` + `1,793M` 语言主干，共 `~2.13B` 参数） |
| **动作专家头 (Action Expert Head)** | 16-Layer Flow-Matching `DiT-B` (`hidden_size = 1024`, `cross_attention_dim = 2048`, `~356.8M` 可训练参数含投影层) |
| **双视角输入与分辨率** | `observation.images.top` + `observation.images.wrist` 统一缩放至 `224×224`，每视角压缩为 `32` 个目标视觉 Token |
| **状态与动作归一化** | 分位数归一化 `QUANTILES [q01, q99]` (`wipes_alicia_q99`, [`dataset_statistics.json`](./dataset_statistics.json))，预测视野 `action_horizon = 30` (`1.0s @ 30 Hz`) |
| **最佳训练配方 (`FixQwen`)** | 冻结完整 `qwen_vl_interface`，仅训练 `action_model` (`lr = 1e-4`, `BS = 4×8 = 32`, `10,000` steps，单卡 RTX 4090 耗时 **`3h 13m`，提速 4.5 倍**) |
| **单块推理延迟 (Chunk Latency)** | **`115.0 ms`** on RTX 4090 (`8.7 Hz`) / **`491.3 ms`** on RTX 3080 Laptop (`2.0 Hz`) |

---

## 2. 纵向对比一：真机闭环时序平滑对比演示 (`FixQwen 10k` — Unsmoothed vs. Quintic $C^2$ Blender)

左半部分为 **无平滑原始 DiT-B 块执行 (`eval_starvla_fixqwen_no_smooth`)**，右半部分为 **接入 `RecedingHorizonTrajectoryBlender` 五次多项式 $C^2$ 边界锚定与块内高斯滤波 (`eval_starvla_fixqwen_smooth`)**：

<div align="center">
  <img src="rollout_comparison_starvla.gif" width="100%" alt="StarVLA Unsmoothed vs Quintic C2 Smoothed Real-Robot Rollout"/>
  <p><em>图 1：StarVLA (FixQwen 10k) 真机闭环同步对比动图（左：无平滑开环块执行，DiT-B 采样存在块内锯齿抖动 SignFlip=54.9%；右：Quintic C² 平滑器消除 97% Jerk 并将 SignFlip 压至 0.3%，实现极度柔顺的抓取与绿盘居中放置）。</em></p>
</div>

---

## 3. 纵向对比二：三路 VLM 主干冻结策略消融 (`Unfrozen` vs. `FreezeViT` vs. `FixQwen`)

在单任务、小样本真机数据集（`67 episodes, 24,535 frames`）上，针对 `2.5B` 级大模型 `StarVLA (Qwen3.5-2B + DiT-B)` 极易发生的单指令/单场景过拟合问题，我们在单卡 `NVIDIA RTX 4090` 上完成了三路严格对照消融：

1. **Run A (`Unfrozen Full Model, 20k steps`)**：放开全部 `Qwen3.5-2B`（`visual` + `LLM`, `lr=1e-5`）与 `DiT-B`（`lr=1e-4`）联合训练（`BS=1×32=32`，需开启梯度检查点）。
2. **Run B (`FreezeViT Only, 10k steps`)**：仅冻结视觉编码器 `qwen_vl_interface.model.model.visual`，放开 `1.8B LLM`（`lr=1e-5`）与 `DiT-B`（`lr=1e-4`）联合微调（`BS=1×32=32`）。
3. **Run C (`FixQwen — Freeze Full Qwen3.5-2B VLM, 10k steps, Best ✅`)**：冻结完整 `qwen_vl_interface`（`ViT + LLM` 全部冻结，对齐 NVIDIA Isaac `GR00T N1.5` 与 `SmolVLA` 专家训练范式），仅训练 `356.8M` 的 `DiT-B` 动作头与跨模态投影层（`BS=4×8=32`）。

<div align="center">
  <img src="starvla_3way_freeze_ablation.png" width="100%" alt="StarVLA 3-Way VLM Freezing Ablation"/>
  <p><em>图 2：StarVLA 三路冻结策略消融结果（左：DiT Flow-Matching 损失收敛曲线；中：单卡 RTX 4090 训练速度与 10k 步总耗时对比；右：三路策略在真机闭环与开环评测中的综合表现对比）。</em></p>
</div>

| 对照实验组 (Ablation Variant) | 冻结模块 (`freeze_modules`) | 可训练参数量 | 单步耗时 / 速度 (RTX 4090) | 10k 步总耗时 | 10k 步 DiT Loss | 开环 6轴 MAE | 真机闭环表现与机理分析 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run A: Unfrozen Full VLM** | 无 (`""`) | `~2,487M` (100%) | 5.11 s/it (`0.20 it/s`) | 860 min (`14.3h`) | 0.0053 | 0.354° *(过拟合)* | **0% 失败** (`eval_starvla_no_smooth`：VLM 视觉与语言表征在单场景上严重退化，真机下探偏离抓空) |
| **Run B: FreezeViT Only** | `qwen_vl_interface.model.model.visual` | `~2,150M` (86%) | 4.53 s/it (`0.22 it/s`) | 772 min (`12.9h`) | 0.0081 | 0.543° | **成功抓放但收尾不彻底** (`eval_starvla_smooth`：能抓起湿巾放入绿盘，但 `1.8B LLM` 在单一文本指令上微调导致末段夹爪仅张开至 `404`) |
| **Run C: FixQwen (Best ✅)** | **`qwen_vl_interface` (Full VLM)** | **`~357M` (14%)** | **1.13 s/it (`0.88 it/s`)** | **193 min (`3.2h`, 快 4.5×)** | **0.0065** | **0.639°** | **100% 完美成功** (`eval_starvla_fixqwen_{smooth,no_smooth}`：完整保留预训练图文空间表征，湿巾精准居中入盘且末段夹爪完全张开至 `996` 并抬升撤离) |

---

## 4. 纵向对比三：真机闭环轨迹与平滑度定量消融 (`RecedingHorizonTrajectoryBlender`)

由于 `StarVLA` 的 `DiT-B` 动作头从标准高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$ 出发执行 10 步 Euler ODE 采样，原始输出块内部相邻时间步之间存在高频采样抖动（无平滑 `eval_starvla_fixqwen_no_smooth` 的速度符号翻转率高达 **`54.9%`**，平均加加速度 `MeanJerk = 0.7096 °/step³`）。接入 `RecedingHorizonTrajectoryBlender` 后，真机平滑度指标获得数量级提升：

<div align="center">
  <img src="starvla_smooth_vs_no_smooth.png" width="100%" alt="StarVLA Smooth vs No-Smooth Trajectory Ablation"/>
  <p><em>图 3：StarVLA (FixQwen 10k) 六轴真机轨迹、步间动作变化 ||Δa_t||₂ 以及四维平滑度指标对比（No-Smooth vs. Quintic C² Blender）。</em></p>
</div>

| 评测配置 (Evaluation Mode) | 轨迹帧数 $N$ | 步间变化均值 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 95分位 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 平均角加速度 (°/step²) $\downarrow$ | 平均加加速度 Jerk (°/step³) $\downarrow$ | 速度符号翻转率 (SignFlip) $\downarrow$ | 最大单步跳变 (MaxJump) $\downarrow$ | 真机放置表现 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **StarVLA (`Unfrozen`) 无平滑 (`eval_starvla_no_smooth`)** | 511 | 0.0144 | 0.0272 | 0.4335 | 0.7894 | 60.6% | 4.10° | 0% (VLM 过拟合导致抓空) |
| **StarVLA (`FreezeViT 10k`) + Quintic $C^2$ (`eval_starvla_smooth`)** | 469 | 0.0053 | 0.0151 | 0.0237 | 0.0208 | 1.2% | 1.07° | 成功入盘 (末段夹爪半开 `404`) |
| **StarVLA (`FixQwen 10k`) 无平滑 (`eval_starvla_fixqwen_no_smooth`)** | 499 | 0.0137 | 0.0260 | 0.3893 | 0.7096 | 54.9% | 3.15° | 100% 成功 (伴随块内高频抖动) |
| **StarVLA (`FixQwen 10k`) + Quintic $C^2$ (`eval_starvla_fixqwen_smooth`, Best ✅)** | **470** | **0.0060 (-56%)** | **0.0157 (-40%)** | **0.0212 (-95%)** | **0.0179 (-97%)** | **0.3%** | **0.88° (-72%)** | **100% 完美居中入盘 (全场最低跳变 `0.88°`)** |

---

## 5. 黄金抓取段开环跟踪误差分项 (Open-Loop Tracking Breakdown)

在 `wipes-to-green-plate-01` 黄金抓取段（Frames 150–300, 5.0s @ 30 Hz）上的各关节开环误差对比：

| 模型变体 (StarVLA Variant) | Joint 1 (Base) | Joint 2 (Shoulder) | Joint 3 (Elbow) | Joint 4 (Forearm) | Joint 5 (Wrist Pitch) | Joint 6 (Wrist Roll) | **6轴平均 MAE** | **6轴 RMSE** | **夹爪 MAE (0–1000)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FixQwen (10k, Full VLM Frozen — Best Real-Robot ✅)** | **0.515°** | **0.699°** | **0.969°** | **0.533°** | **0.691°** | **0.429°** | **0.639°** | **0.889°** | **16.90** |
| **FreezeViT (10k, Visual Tower Frozen)** | 0.487° | 0.633° | 0.645° | 0.416° | 0.521° | 0.555° | 0.543° | 0.764° | 17.18 |
| **Unfrozen Full VLM (20k, Overfit)** | 0.345° | 0.407° | 0.476° | 0.298° | 0.285° | 0.313° | 0.354° *(过拟合)* | 0.462° | 11.31 |

---

## 6. 本目录文件与复现命令 (Files & Reproducibility)

- [`run_wipes_fixqwen_4090.sh`](./run_wipes_fixqwen_4090.sh) & [`starvla_wipes_qwen3_5_2b_gr00t_fixqwen.yaml`](./starvla_wipes_qwen3_5_2b_gr00t_fixqwen.yaml)：冻结完整 `Qwen3.5-2B` VLM 训练脚本与配置（10,000 steps，`3h 13m` on RTX 4090）
- [`run_wipes_freeze_vit_4090.sh`](./run_wipes_freeze_vit_4090.sh) & [`starvla_wipes_qwen3_5_2b_gr00t_freeze_vit.yaml`](./starvla_wipes_qwen3_5_2b_gr00t_freeze_vit.yaml)：仅冻结视觉塔 `visual` 对照训练脚本与配置
- [`dataset_statistics.json`](./dataset_statistics.json)：`wipes_alicia_q99` 分位数与均值方差归一化统计量
- [`eval_starvla_smooth.sh`](./eval_starvla_smooth.sh)：真机闭环评测（`FixQwen 10k` + `Quintic C² Blender`）
- [`eval_starvla_no_smooth.sh`](./eval_starvla_no_smooth.sh)：真机闭环评测（`FixQwen 10k` 无平滑对照基线）
