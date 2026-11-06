# TurboVLA (1-Step MeanFlow + Frozen Real-World DINOv3) — Alicia-D 真机纵向深入消融 (`分比`)

[⬅️ 返回跨策略横向总览 (Back to Main Benchmark README)](../README.md)

---

## 1. 模型架构与配置概览

| 属性项 | 规格配置 |
| :--- | :--- |
| **生成范式** | 单步均值流场生成 **1-Step MeanFlow (`1-NFE` JVP Velocity Distillation)** |
| **视觉与语言主干** | **冻结真实世界预训练 `DINOv3-ViT-B/16` (`LVD-1689M`)** + `BERT-base-uncased` + 6 层多模态融合增强器 |
| **总参数量 / 可训练参数量** | 总计 `~216M`；冻结 `DINOv3` 与 `BERT` 后**可训练参数仅 `20.6M (9.5%)`** |
| **跨本体状态对齐** | 7 维 Alicia-D 状态末尾补零对齐至 LIBERO 预训练模型的 8 维输入 (`[J1..J6, Gripper, 0.0]`) |
| **预测视野 (Chunk Horizon $K$)** | `30` 步（从 `LIBERO` 预训练的 `12` 个动作查询向量通过线性插值平滑拉伸至 `30` 步） |
| **最优训练配置 (`Warm-Full`)** | `--freeze_vision --keep_real_dinov3 --load_mode warm_full`，`12,500` 步，有效批大小 `64`，单卡 `RTX 4090` 耗时 **`36.3 分钟`** |
| **单块推理延迟 (Chunk Latency)** | **`16.8 ms`** on RTX 4090 (`59.5 Hz`) / **`24.6 ms`** on RTX 3080 Laptop (`40.7 Hz`) — **严格低于 `30 Hz` 控制环单帧预算 (`33.3 ms`)** |

---

## 2. 真机闭环平滑对比实拍（`Warm-Full 12.5k` 无平滑 vs. 五次多项式 $C^2$ 边界锚定平滑）

在 Alicia-D 真机 30 Hz 闭环测试中，左半部分为 **无平滑开环块执行 (`eval_turbovla_warm_full_no_smooth`, `n_action_steps = 15`)**，右半部分为 **启用 `RecedingHorizonTrajectoryBlender` 五次多项式 $C^2$ 边界锚定 + 块内高斯滤波 (`eval_turbovla_warm_full_smooth`, $\sigma=1.8, K=12$)**。两组实验真机表现均为**成功**，其中启用五次多项式 $C^2$ 平滑后彻底消除了 1-NFE 并行动作查询 (`[Q_0..Q_29]`) 之间的锯齿微颤与块切换阶跃：

<div align="center">
  <img src="rollout_comparison_turbovla.gif" width="100%" alt="TurboVLA Warm-Full 12.5k 无平滑 vs 五次多项式 C2 平滑真机对比"/>
  <p><em>图 1：TurboVLA (Warm-Full 12.5k) 真机 30 Hz 闭环同步对比动图（左：无平滑分块执行，真机表现成功但存在并行 Query 锯齿感；右：启用五次多项式 C² 边界锚定与高斯平滑，三阶加加速度 Jerk 降低 96%，丝滑完成抓取与入盘）。</em></p>
</div>

---

## 3. 纵向对比一：跨本体迁移、冻结视觉主干与选择性热启动消融 (`LIBERO` $\rightarrow$ `Alicia-D`)

针对小样本单场景真机数据集（67 条轨迹，`24,535` 帧）上跨本体迁移的核心问题——**（1）视觉编码器是否应当冻结？（2）预训练 `LIBERO` 检查点的哪些模块具有正向迁移价值？**——我们在单卡 RTX 4090 上完成了五路对照消融实验：

1. **Run B（从头训练 `Scratch 20k`，未冻结 ViT）**：`DINOv3-ViT-B/16` 全参数放开微调 (`lr_dino=3.5e-5`)，融合增强器与动作头随机初始化。
2. **Run A（`Sim-ViT Warm-Start 20k`，未冻结仿真 ViT）**：从 `LIBERO 40k` 全量加载（含被 MuJoCo 合成图像微调过的 `DINOv3`）并继续放开视觉主干微调。
3. **Run C（`Frozen-DINOv3 + Warm-Full 8k`）**：保留并冻结 Meta 官方真实世界 `DINOv3 (LVD-1689M)` (`--freeze_vision --keep_real_dinov3`)，从 `LIBERO 40k` 热启动融合增强器与 MeanFlow 动作头，训练 `8,000` 步（`23.0 分钟`）。
4. **Run D（`Frozen-DINOv3 + Warm-Body 12.5k`，仅热启动编码器）**：冻结真实世界 `DINOv3`，仅从 `LIBERO 40k` 热启动 6 层 VLA 融合增强器 (`--load_mode warm_body`)，而将 1-NFE MeanFlow 动作头 (`action_head`) 与 `action_queries` 随机初始化，训练 `12,500` 步（`36.2 分钟`）。
5. **Run E（`Frozen-DINOv3 + Warm-Full 12.5k`，全量热启动 — 最优配置）**：冻结真实世界 `DINOv3`，同时从 `LIBERO 40k` 热启动 6 层融合增强器、1-NFE MeanFlow 动作头与线性插值动作查询 (`--load_mode warm_full`)，训练 `12,500` 步（`36.3 分钟`）。

<div align="center">
  <img src="turbovla_3way_transfer_ablation.png" width="100%" alt="TurboVLA 跨本体迁移与选择性热启动消融对比"/>
  <p><em>图 2：TurboVLA 五路跨本体迁移与选择性热启动消融实验对比：L1 动作损失收敛曲线（左）、开环 6 轴 MAE 与真机表现（中）、RTX 4090 训练耗时与吞吐率对比（右）。</em></p>
</div>

| 实验配置 | 视觉主干 (`DINOv3-ViT-B/16`) | 多模态融合层与动作头初始化 | 训练速度 | 总训练耗时 | 最终 L1 Loss | 开环 6轴 MAE | 真机表现 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Run B: `Scratch` (20k)** | 未冻结 (`lr=3.5e-5`) | 随机初始化 | 1.39 step/s | 162.8 分钟 | 0.0059 | 0.145° | 失败（单场景背景过拟合） |
| **Run A: `Sim-ViT Warm` (20k)** | 未冻结 (`LIBERO` 仿真微调权重) | `LIBERO 40k` 全量加载 | 1.39 step/s | 162.4 分钟 | 0.0053 | 0.088° | 失败（仿真视觉特征偏移） |
| **Run C: `Frozen-ViT Warm-Full` (8k)** | **冻结真实世界 `LVD-1689M`** | `LIBERO 40k` 全量热启动 | **5.79 step/s** | **23.0 分钟** | 0.0340 | 0.606° | **成功** |
| **Run D: `Frozen-ViT Warm-Body` (12.5k)** | **冻结真实世界 `LVD-1689M`** | 仅热启动融合层（动作头随机初始化） | 2.72 step/s | 36.2 分钟 | 0.0131 | 0.171° | **成功**（无平滑抖动较大） |
| **Run E: `Frozen-ViT Warm-Full` (12.5k)** | **冻结真实世界 `LVD-1689M`** | **`LIBERO 40k` 融合层 + 动作头全热启动** | **2.72 step/s** | **36.3 分钟** | **0.0110** | **0.166°** | **成功（最优配置）** |

### 核心机制剖析
1. **保留并冻结真实世界 `DINOv3` (`--freeze_vision --keep_real_dinov3`)**：放开 `86M` 参数的 `DINOv3` 在 67 条单场景轨迹上训练会使模型记忆背景环境亮度（`Run A/B` 虽取得 `0.088°~0.145°` 开环误差但真机闭环失败）；冻结真实世界预训练 `DINOv3` 后，`Run C`、`Run D` 与 `Run E` 在真机闭环下**全部成功完成抓取与放置**，且训练耗时从 `163 分钟` 大幅缩短至 **`23~36 分钟`**。
2. **预训练 MeanFlow 动作头热启动的平滑优势 (`Warm-Full` vs. `Warm-Body`)**：对比 `Run D (Warm-Body)` 与 `Run E (Warm-Full)`，两者在真机上均成功完成任务，但复用 `LIBERO 40k` 预训练的 1-NFE MeanFlow 动作头与插值查询向量 (`Warm-Full`) 提供了良好的连续速度场先验，使真机无平滑下的三阶加加速度 `MeanJerk` 从 `1.1857 °/step³` 降至 **`0.4067 °/step³`（降低 66%）**，启用平滑后进一步降至 **`0.0151 °/step³`**。

---

## 4. 纵向对比二：真机闭环五次多项式 $C^2$ 平滑消融（六组真机实测全对比）

<div align="center">
  <img src="turbovla_smooth_vs_no_smooth.png" width="100%" alt="TurboVLA Warm-Full 12.5k 平滑 vs 无平滑轨迹消融"/>
  <p><em>图 3：TurboVLA (Warm-Full 12.5k) 六轴真机轨迹、步间动作变化 ||Δa_t||₂ 及平滑度指标对比（eval_turbovla_warm_full_no_smooth vs. eval_turbovla_warm_full_smooth）。</em></p>
</div>

| 策略变体与闭环执行模式 | 真机评测目录 (`eval_runs/local/`) | 轨迹帧数 $N$ | 步间变化均值 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 95分位 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 平均角加速度 (°/step²) $\downarrow$ | 平均加加速度 Jerk (°/step³) $\downarrow$ | 速度符号翻转率 (SignFlip) $\downarrow$ | 最大单步跳变 (MaxJump) $\downarrow$ | 真机表现 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Frozen-ViT (8k)` 无平滑** | `eval_turbovla_no_smooth` | 705 (`23.5s`) | 0.0111 | 0.0222 | 0.2829 | 0.5103 | 42.3% | 5.88° | **成功** |
| **`Frozen-ViT (8k)` + 五次多项式 $C^2$** | `eval_turbovla_smooth` | 711 (`23.7s`) | **0.0058 (-48%)** | **0.0157 (-29%)** | **0.0253 (-91%)** | **0.0172 (-97%)** | **1.1%** | **1.11° (-81%)** | **成功** |
| **`Warm-Body (12.5k)` 无平滑** | `eval_turbovla_warm_body_no_smooth` | 709 (`23.6s`) | 0.0214 | 0.0376 | 0.6477 | 1.1857 | 61.3% | 5.21° | **成功** |
| **`Warm-Body (12.5k)` + 五次多项式 $C^2$** | `eval_turbovla_warm_body_smooth` | 709 (`23.6s`) | **0.0087 (-59%)** | **0.0207 (-45%)** | **0.0458 (-93%)** | **0.0325 (-97%)** | **3.1%** | **1.44° (-72%)** | **成功** |
| **`Warm-Full (12.5k)` 无平滑** | `eval_turbovla_warm_full_no_smooth` | 709 (`23.6s`) | 0.0102 | 0.0326 | 0.2191 | 0.4067 | 35.8% | 5.98° | **成功** |
| **`Warm-Full (12.5k)` + 五次多项式 $C^2$** | `eval_turbovla_warm_full_smooth` | **710 (`23.7s`)** | **0.0052 (-49%)** | **0.0194 (-40%)** | **0.0224 (-90%)** | **0.0151 (-96%)** | **1.2%** | **1.31° (-78%)** | **成功** |

---

## 5. 黄金抓取段开环跟踪误差分项（Frame 150–300，5.0s @ 30 Hz）

| TurboVLA 变体 | Joint 1 (底座) | Joint 2 (肩部) | Joint 3 (肘部) | Joint 4 (前臂) | Joint 5 (腕俯仰) | Joint 6 (腕横滚) | **6轴平均 MAE** $\downarrow$ | **6轴 RMSE** $\downarrow$ | **夹爪 MAE (0–1000)** $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Frozen-DINOv3 Warm-Full (8k)`** | 0.651° | 0.884° | 0.712° | 0.542° | 0.485° | 0.362° | 0.606° | 0.908° | 20.20 |
| **`Frozen-DINOv3 Warm-Body (12.5k)`** | 0.145° | 0.175° | 0.222° | 0.147° | 0.181° | 0.155° | 0.171° | 0.223° | 7.36 |
| **`Frozen-DINOv3 Warm-Full (12.5k)`** | **0.118°** | **0.168°** | **0.216°** | **0.150°** | **0.170°** | **0.178°** | **0.166°** | **0.217°** | **6.53** |

---

## 6. 本目录文件与复现命令

- [`train_turbovla_wipes_warm_ab.sh`](./train_turbovla_wipes_warm_ab.sh) & [`train_turbovla_wipes_selective.sh`](./train_turbovla_wipes_selective.sh)：`Warm-Full` 与 `Warm-Body` 选择性热启动训练脚本（12,500 steps）
- [`train_turbovla_wipes_freeze_vit.sh`](./train_turbovla_wipes_freeze_vit.sh)：冻结真实世界 `DINOv3` + `LIBERO 40k` 快速热启动训练脚本（8,000 steps）
- [`eval_turbovla_warm_full_smooth.sh`](./eval_turbovla_warm_full_smooth.sh) & [`eval_turbovla_warm_full_no_smooth.sh`](./eval_turbovla_warm_full_no_smooth.sh)：`Warm-Full (12.5k)` 真机闭环评测脚本（平滑 vs. 无平滑）
- [`eval_turbovla_warm_body_smooth.sh`](./eval_turbovla_warm_body_smooth.sh) & [`eval_turbovla_warm_body_no_smooth.sh`](./eval_turbovla_warm_body_no_smooth.sh)：`Warm-Body (12.5k)` 真机闭环评测脚本（平滑 vs. 无平滑）
- [`eval_turbovla_smooth.sh`](./eval_turbovla_smooth.sh) & [`eval_turbovla_no_smooth.sh`](./eval_turbovla_no_smooth.sh)：`Frozen-DINOv3 (8k)` 真机闭环评测脚本（平滑 vs. 无平滑）
