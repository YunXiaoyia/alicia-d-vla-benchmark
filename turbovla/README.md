# TurboVLA (1-NFE MeanFlow VLA) — Alicia-D Real-Robot Deep Dive

[⬅️ 返回总览对比 (Back to Main Benchmark README)](../README.md)

---

## 1. 模型架构与跨本体适配规格 (Architecture & Cross-Embodiment Spec)

| 属性 (Property) | 规格配置 (Specification) |
| :--- | :--- |
| **生成范式 (Generative Paradigm)** | **1-Step MeanFlow (`1-NFE`)** 单步流匹配解码器（无需多步 ODE 迭代） |
| **视觉与语言主干 (Backbones)** | `DINOv3-ViT-B/16` (`LVD-1689M`) + Frozen `BERT-base-uncased` |
| **多模态融合与解码 (Fusion & Head)** | 6-Layer Bi-Directional Cross-Attention Feature Enhancer + Parallel Action Query Decoder (`[Q_0..Q_29]`) |
| **参数量 (Total Parameters)** | `~216M`（全量微调可训 `106.6M`；冻结视觉与语言后仅训 `20.9M` 融合层与动作头） |
| **跨本体状态与视野对齐** | `7D -> 8D` 零填充对齐 `LIBERO` 预训练 `state_encoder`；`action_queries` 线性插值 `[12, 256] -> [30, 256]` |
| **单块推理延迟 (Chunk Latency)** | **`16.8 ms`** on RTX 4090 (`59.5 Hz`) / **`24.6 ms`** on RTX 3080 Laptop (`40.7 Hz`)，严格低于 `30 Hz` (`33.3 ms`) 控制周期预算 |

---

## 2. 纵向对比一：真机闭环时序平滑对比演示 (Unsmoothed 1-NFE vs. Quintic $C^2$ Receding-Horizon Blender)

左半部分为 **无平滑 1-NFE 原始块执行 (`eval_turbovla_warm_full_no_smooth`)**，右半部分为 **接入 `RecedingHorizonTrajectoryBlender` 五次多项式 $C^2$ 锚定与块内因果高斯平滑 (`eval_turbovla_warm_full_smooth`)**：

<div align="center">
  <img src="rollout_comparison_turbovla.gif" width="100%" alt="TurboVLA Unsmoothed vs Quintic C2 Smoothed Real-Robot Rollout"/>
  <p><em>图 1：TurboVLA (Warm-Full 12.5k) 真机闭环同步对比动图（左：无平滑原始块执行，存在并行 Query 锯齿抖动与块边界阶跃；右：Quintic C² 平滑器消除 96% Jerk，实现极度柔顺的精准抓取、入盘放置与抬臂撤离）。</em></p>
</div>

---

## 3. 纵向对比二：四路跨本体热启动与动作头/视觉主干消融 (`LIBERO` $\rightarrow$ `Alicia-D`)

围绕 **“跨本体预训练权重该如何迁移至真机小样本数据集（`67 episodes, 24,535 frames`）”**，我们在 `NVIDIA RTX 4090` 与真实 `Alicia-D` 机械臂上完成了四路对照实验：

1. **Run D (`From Scratch, 20k steps`)**：不加载 `LIBERO` 融合层与动作头，仅从 `DINOv3` + `BERT` 从头训练。
2. **Run B (`Warm-Body: Backbone Warm-Start + Cold-Start Action Head, 12.5k steps`)**：加载 `turbovla_libero.pth` 的视觉/语言/融合层权重，但通过 `--reinit_action_head` 冷启动重置 7 维动作头。
3. **Run C (`Frozen-DINOv3: Frozen Real-World DINOv3 + LIBERO Action Head, 8k steps`)**：启用 `--keep_real_dinov3 --freeze_vision`，冻结 `LVD-1689M` 真实世界 `DINOv3`，仅热启动并微调融合层与 `LIBERO` 动作头（耗时仅 **`23.0 min`，提速 7.1 倍**）。
4. **Run A (`Warm-Full: Full LIBERO 672-Tensor Warm-Start, 12.5k steps — Best ✅`)**：完整加载 `turbovla_libero.pth` 全部 672 个张量（含 7 维同构关节空间动作头，`action_queries` 由 12 步线性插值至 30 步），配合在线顶视相机像素级配准 (`align_top_camera.py`) 进行闭环部署。

<div align="center">
  <img src="turbovla_3way_transfer_ablation.png" width="100%" alt="TurboVLA 4-Way Cross-Embodiment Transfer Ablation"/>
  <p><em>图 2：TurboVLA 四路跨本体迁移消融结果（左：L1 动作损失收敛曲线；中：训练耗时与迭代速度对比；右：四路模型在真机闭环评测中的步间平滑度与 Jerk 对比）。</em></p>
</div>

| 对照实验组 (Ablation Variant) | 视觉主干 (`DINOv3-ViT-B/16`) | 动作头 (`action_head`) 初始化 | 训练步数 / 耗时 (RTX 4090) | 最终 L1 Loss | 开环 6轴 MAE | 开环夹爪 MAE | 真机闭环表现与关键结论 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Run D: From Scratch** | 解冻 (`lr=3.5e-5`) | 随机冷启动 | 20,000 (`162.8 min`) | 0.0059 | 0.124° *(过拟合)* | 4.05 | **0% 失败** (`eval_turbovla_scratch_*`：缺乏跨本体先验，真机闭环无法完成有效抓取) |
| **Run B: Warm-Body (`--reinit_action_head`)** | 解冻 (`lr=5e-5`) | **随机冷启动 (`--reinit`)** | 12,500 (`~102 min`) | 0.0094 | 0.248° | 8.39 | **能抓放入盘但收尾迟滞** (`eval_turbovla_warm_body_*`：丢弃同维 7 轴动作头先验导致无平滑 Jerk 恶化 **2.9 倍** `1.1857 °/step³`，入盘后抬臂撤离缓慢 `end_J2=-21.8°`) |
| **Run C: Frozen-DINOv3 (`--freeze_vision`)** | **冻结真实 `LVD-1689M`** | **加载 LIBERO 7D 动作头** | **8,000 (`23.0 min`, 快 7.1×)** | 0.0340 | 0.606° | 20.20 | **100% 真机成功** (`eval_turbovla_smooth`：抗光照漂移能力最强，湿巾精准居中入盘并干净撤离 `end_J2=-16.5°`) |
| **Run A: Warm-Full (Full 672 Tensors, Best ✅)** | 解冻 (`lr=5e-5`) | **加载 LIBERO 7D 动作头** | 12,500 (`~102 min`) | **0.0079** | **0.166°** | **6.53** | **100% 真机成功** (`eval_turbovla_warm_full_smooth`：在相机视角配准下兼顾最高轨迹跟踪精度 `0.166°` 与全场最低闭环加加速度 `Jerk = 0.0151 °/step³`，入盘后干脆撤离 `end_J2=-14.9°`) |

---

## 4. 纵向对比三：真机闭环时序平滑消融 (`RecedingHorizonTrajectoryBlender` 全变体横评)

`TurboVLA` 通过 30 个并行 Action Query (`[Q_0..Q_29]`) 一步生成完整动作块 (`1-NFE`)，相邻 Query 间缺乏 ODE 积分连续性约束。接入四级因果平滑器 `RecedingHorizonTrajectoryBlender`（块内 $\sigma=1.8$ 高斯滤波 + 前半块防回退夹紧 + **五次多项式最小加加速度锚定 $s(\tau)=10\tau^3-15\tau^4+6\tau^5$** + 搬运期夹爪防抖锁存）后，三组真机成功变体的闭环平滑度对比如下：

<div align="center">
  <img src="turbovla_smooth_vs_no_smooth.png" width="100%" alt="TurboVLA Smooth vs No-Smooth Trajectory Ablation"/>
  <p><em>图 3：TurboVLA (Warm-Full 12.5k) 六轴真机轨迹、步间动作变化 ||Δa_t||₂ 以及四维平滑度指标对比（No-Smooth vs. Quintic C² Blender）。</em></p>
</div>

| 评测配置 (Evaluation Run) | 轨迹帧数 $N$ | 步间变化均值 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 95分位 $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 平均角加速度 (°/step²) $\downarrow$ | 平均加加速度 Jerk (°/step³) $\downarrow$ | 速度符号翻转率 (SignFlip) $\downarrow$ | 最大单步跳变 (MaxJump) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Warm-Body (12.5k)` 无平滑 (`eval_turbovla_warm_body_no_smooth`)** | 709 | 0.0214 | 0.0376 | 0.6477 | 1.1857 | 61.3% | 5.21° |
| **`Warm-Body (12.5k)` + Quintic $C^2$ (`eval_turbovla_warm_body_smooth`)** | 709 | **0.0087 (-59%)** | **0.0207 (-45%)** | **0.0458 (-93%)** | **0.0325 (-97%)** | **3.1%** | **1.44° (-72%)** |
| **`Frozen-DINOv3 (8k)` 无平滑 (`eval_turbovla_no_smooth`)** | 705 | 0.0111 | 0.0222 | 0.2829 | 0.5103 | 42.3% | 5.88° |
| **`Frozen-DINOv3 (8k)` + Quintic $C^2$ (`eval_turbovla_smooth`)** | 711 | **0.0058 (-48%)** | **0.0157 (-29%)** | **0.0253 (-91%)** | **0.0172 (-97%)** | **1.1%** | **1.11° (-81%)** |
| **`Warm-Full (12.5k)` 无平滑 (`eval_turbovla_warm_full_no_smooth`)** | 709 | 0.0102 | 0.0326 | 0.2191 | 0.4067 | 35.8% | 5.98° |
| **`Warm-Full (12.5k)` + Quintic $C^2$ (`eval_turbovla_warm_full_smooth`, Best ✅)** | **710** | **0.0052 (-49%)** | **0.0194 (-40%)** | **0.0224 (-90%)** | **0.0151 (-96%)** | **1.2%** | **1.31° (-78%)** |

---

## 5. 黄金抓取段开环跟踪误差分项 (Open-Loop Tracking Breakdown)

在 `wipes-to-green-plate-01` 黄金抓取段（Frames 150–300, 5.0s @ 30 Hz）上的各关节开环误差对比：

| 模型变体 (TurboVLA Variant) | Joint 1 (Base) | Joint 2 (Shoulder) | Joint 3 (Elbow) | Joint 4 (Forearm) | Joint 5 (Wrist Pitch) | Joint 6 (Wrist Roll) | **6轴平均 MAE** | **6轴 RMSE** | **夹爪 MAE (0–1000)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warm-Full (12.5k, Full 672 Tensors — Best ✅)** | **0.118°** | **0.168°** | **0.216°** | **0.150°** | **0.170°** | **0.178°** | **0.166°** | **0.217°** | **6.53** |
| **Warm-Body (12.5k, Reinit Action Head)** | 0.169° | 0.255° | 0.348° | 0.208° | 0.275° | 0.230° | 0.248° | 0.329° | 8.39 |
| **Frozen-DINOv3 (8k, Frozen Real ViT)** | 0.444° | 0.613° | 0.664° | 0.501° | 0.631° | 0.784° | 0.606° | 0.908° | 20.20 |

---

## 6. 本目录文件与复现命令 (Files & Reproducibility)

- [`train_turbovla_wipes_warm_ab.sh`](./train_turbovla_wipes_warm_ab.sh)：`Warm-Full` vs. `Warm-Body` 单变量对照训练队列脚本（12,500 steps）
- [`train_turbovla_wipes_freeze_vit.sh`](./train_turbovla_wipes_freeze_vit.sh) & [`train_turbovla_wipes_selective.sh`](./train_turbovla_wipes_selective.sh)：冻结真实世界 `DINOv3` + `LIBERO` 热启动训练脚本（8,000 steps，23 min on RTX 4090）
- [`eval_turbovla_warm_full_smooth.sh`](./eval_turbovla_warm_full_smooth.sh) & [`eval_turbovla_warm_full_no_smooth.sh`](./eval_turbovla_warm_full_no_smooth.sh)：`Warm-Full (12.5k)` 真机闭环评测脚本
- [`eval_turbovla_warm_body_smooth.sh`](./eval_turbovla_warm_body_smooth.sh) & [`eval_turbovla_warm_body_no_smooth.sh`](./eval_turbovla_warm_body_no_smooth.sh)：`Warm-Body (12.5k)` 真机闭环评测脚本
- [`eval_turbovla_smooth.sh`](./eval_turbovla_smooth.sh) & [`eval_turbovla_no_smooth.sh`](./eval_turbovla_no_smooth.sh)：`Frozen-DINOv3 (8k)` 真机闭环评测脚本
