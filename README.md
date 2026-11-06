# Alicia-D 6-DoF Vision-Language-Action (VLA) Real-Robot Benchmark

<div align="center">

[![Framework: LeRobot v0.4.4](https://img.shields.io/badge/Framework-LeRobot%20v0.4.4-blue.svg)](https://github.com/huggingface/lerobot)
[![Hardware: Alicia-D 6-DoF](https://img.shields.io/badge/Hardware-Alicia--D%206--DoF%20Follower-00b894.svg)](#1-hardware--task-setup)
[![Models: ACT | SmolVLA | π0.5 | StarVLA | TurboVLA](https://img.shields.io/badge/Models-ACT%20%7C%20SmolVLA%20%7C%20%CF%800.5%20%7C%20StarVLA%20%7C%20TurboVLA-fdcb6e.svg)](#3-experimental-results--cross-model-benchmark-analysis)
[![Control Rate: 30 Hz](https://img.shields.io/badge/Real--Time%20Control-30%20Hz-e17055.svg)](#32-inference-latency--computational-efficiency)

**A unified, reproducible real-world manipulation benchmark comparing classic CVAE policies (`ACT`), compact Flow-Matching VLAs (`SmolVLA`), 2B–3B+ foundation VLAs (`StarVLA`, `π0.5`), and 1-Step MeanFlow VLAs (`TurboVLA`) on a low-cost 6-DoF robotic arm.**

</div>

---

## Live Closed-Loop Real-Robot Rollouts (30 Hz — Best Configuration per Policy Family)

Synchronized **Global Top View + Wrist Eye-in-Hand View** closed-loop autonomous rollouts on physical Alicia-D hardware for the task *"Pick up the pack of cleaning wipes and place it on the green plate"*:

| **TurboVLA (Warm-Full 12.5k, Quintic $C^2$ Smooth)** | **StarVLA (FixQwen 10k, Quintic $C^2$ Smooth)** |
| :---: | :---: |
| ![TurboVLA Rollout](assets/demos/rollout_turbovla_smooth.gif) | ![StarVLA Rollout](assets/demos/rollout_starvla_smooth.gif) |
| **$\pi_{0.5}$ (Full 20k, Flow-Matching RTC)** | **SmolVLA (30k, Flow-Matching RTC)** |
| ![pi0.5 Rollout](assets/demos/rollout_pi05_rtc.gif) | ![SmolVLA Rollout](assets/demos/rollout_smolvla_ensemble.gif) |

<div align="center">

| **ACT (50k, Temporal Ensembling $\lambda=0.01$)** |
| :---: |
| ![ACT Rollout](assets/demos/rollout_act50_ensemble.gif) |

</div>

---

## Executive Summary (`横向对比 / 总比` — Best Variant of Each Policy Family)

Deploying modern **Vision-Language-Action (VLA)** foundation models onto non-standard, low-cost robotic hardware remains bottlenecked by three fundamental challenges:
1. **Cross-Embodiment Action & Camera Mismatch**: Pretrained foundation models (`π0.5`, `StarVLA`, `SmolVLA`, `TurboVLA`) expect heterogeneous camera slot conventions, action horizons (`10~50` steps), and normalization schemes (`Mean/Std`, `Min/Max [-1, 1]`, or `Quantiles [q01, q99]`).
2. **Inference Latency vs. Multi-Step Denoising Bottlenecks**: Iterative diffusion and flow-matching decoders (`10` Euler ODE steps in `π0.5`, `StarVLA`, and `SmolVLA`) introduce `56–491 ms` chunk generation latency, causing control stalls at chunk boundaries during `30 Hz` closed-loop execution.
3. **Chunk-Boundary Discontinuities & Single-Scene Backbone Overfitting**: Executing open-loop action chunks without cross-chunk temporal blending induces sharp step-to-step velocity jumps (`up to 12.98°/step`) and mechanical vibration; meanwhile, unfreezing large visual/VLM backbones (`DINOv3-ViT-B/16` or `Qwen3.5-2B-VL`) on compact real-robot datasets (`67 episodes`) causes severe overfitting to static ambient lighting.

This repository provides a **complete, end-to-end hardware-software stack** and **rigorous quantitative benchmark** for the **Alicia-D 6-DoF robotic manipulator** (equipped with dual `640×480 @ 30 Hz` global and wrist RGB cameras), evaluating **five representative policy families** (selecting the best configuration of each family for cross-model comparison):

| Policy Family (Best Variant) | Generative Paradigm | Visual & Language Backbone | Action Chunk $K$ | Denoising Steps (NFE) | 6-Joint MAE (°) $\downarrow$ | 6-Joint RMSE (°) $\downarrow$ | Cosine Sim $\uparrow$ | Gripper MAE $\downarrow$ | Chunk Latency (RTX 4090) $\downarrow$ | Max Control Rate $\uparrow$ | Real-Robot Closed-Loop Success |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[ACT (50k)](./act/)** | Conditional VAE (L1) | ResNet-18 (ImageNet) | 50 | **1 (Direct)** | 1.209° | 1.494° | 0.9970 | 14.12 | **6.1 ms** | **163.9 Hz** | **100%** (w/ Temporal Ensemble) |
| **[SmolVLA (30k)](./smolvla/)** | Flow Matching ODE | SmolVLM2-500M | 30 | 10 (Euler) | 0.386° | 0.501° | 0.9998 | 26.43 | 56.3 ms | 17.8 Hz | **100%** (w/ Flow-Matching RTC) |
| **[$\pi_{0.5}$ (Full 20k)](./pi0_5/)** | Flow Matching ODE | PaliGemma-3B + Action Expert | 30 | 10 (Euler) | 0.382° | 0.518° | 0.9997 | 19.23 | 112.0 ms | 8.9 Hz | **100%** (w/ Flow-Matching RTC) |
| **[StarVLA (FixQwen 10k)](./starvla/)** | Flow Matching DiT-B | **Frozen Qwen3.5-2B-VL** + GR00T DiT | 30 | 10 (Euler) | 0.639° | 0.889° | 0.9992 | 16.90 | 115.0 ms | 8.7 Hz | **100%** (w/ Quintic $C^2$ Blender) |
| **[TurboVLA (Warm-Full 12.5k)](./turbovla/)** | **1-Step MeanFlow** | **Frozen Real DINOv3-ViT-B/16** + BERT | 30 | **1 (1-NFE)** | **0.166°** | **0.217°** | **1.0000** | **6.53** | **16.8 ms** | **59.5 Hz** | **100%** (w/ Quintic $C^2$ Blender) |

---

### Per-Model Deep-Dive Sub-Directories (`纵向对比 / 分比` 专页索引)

While this main page presents the **horizontal cross-model comparison (`横向对比 / 总比`)** across the best variant of each of the 5 policy families, each policy family also has a dedicated sub-directory (`纵向对比 / 分比`) containing its **intra-policy ablations**, **Unsmoothed vs. Smoothed Real-Robot Rollout Comparison GIF**, **6-Joint Trajectory & Smoothness Plots**, and **Self-Contained Training/Evaluation Scripts**:

| Model Directory | Dedicated Vertical Report (`分比`) | Side-by-Side Comparison GIF (`No-Smooth` vs. `Smooth`) | Intra-Policy Vertical Ablation Highlights |
| :--- | :--- | :--- | :--- |
| **[`act/`](./act/)** | **[`act/README.md`](./act/README.md)** | [`act/rollout_comparison_act.gif`](./act/rollout_comparison_act.gif) | Open-Loop Chunking (`n=30`) vs. Exponential Temporal Ensembling ($\lambda=0.01$): `-92%` Jerk (`0.5212 -> 0.0408`), `-81%` Max Jump (`12.98° -> 2.45°`) |
| **[`smolvla/`](./smolvla/)** | **[`smolvla/README.md`](./smolvla/README.md)** | [`smolvla/rollout_comparison_smolvla.gif`](./smolvla/rollout_comparison_smolvla.gif) | Flow-Matching RTC vs. No-Smooth + Execution Horizon Ablation (`n_action_steps = 15 vs. 20 vs. 30`) |
| **[`pi0_5/`](./pi0_5/)** | **[`pi0_5/README.md`](./pi0_5/README.md)** | [`pi0_5/rollout_comparison_pi05.gif`](./pi0_5/rollout_comparison_pi05.gif) | `A800 Full FT` vs. `4090 LoRA Merger` + Flow-Matching RTC (`-83%` Chunk-Boundary Jump: `7.71° -> 1.33°`) |
| **[`starvla/`](./starvla/)** | **[`starvla/README.md`](./starvla/README.md)** | [`starvla/rollout_comparison_starvla.gif`](./starvla/rollout_comparison_starvla.gif) | **3-Way VLM Freezing Ablation** (`Unfrozen 20k` vs. `FreezeViT 10k` vs. `FixQwen 10k`: `3.9×` faster, `100%` Real Success) + Quintic $C^2$ Blender (`-97%` Jerk) |
| **[`turbovla/`](./turbovla/)** | **[`turbovla/README.md`](./turbovla/README.md)** | [`turbovla/rollout_comparison_turbovla.gif`](./turbovla/rollout_comparison_turbovla.gif) | **5-Way Cross-Embodiment Transfer & Selective Warm-Start Ablation** (`Scratch` vs. `Sim-ViT` vs. `Frozen-ViT 8k` vs. `Warm-Body` vs. `Warm-Full 12.5k`) + Quintic $C^2$ Blender (`-96%` Jerk) |

---

## 1. Hardware & Task Setup

### 1.1 Alicia-D 6-DoF Manipulator & Dual-Camera Suite

The physical platform consists of an **Alicia-D 6-DoF serial manipulator** with a parallel-jaw servo gripper (`alicia_d_sdk` high-speed serial protocol at `1,000,000 baud`) and a synchronized dual-camera perception system:

- **Kinematic State Space (`observation.state`, $\mathbb{R}^7$)**: 6 revolute joint angles (`joint1.pos` .. `joint6.pos`, in degrees) + 1 continuous gripper position (`gripper.pos` $\in [0, 1000]$, where `0` = fully closed and `1000` = fully open).
- **Action Space (`action`, $\mathbb{R}^7$)**: Absolute target joint angles (`joint1..6.pos` in degrees) + target gripper opening (`gripper.pos` $\in [0, 1000]$) streamed at **30 Hz**.
- **Dual-Camera Perception (`640×480 @ 30 Hz`)**:
  - `observation.images.top` (`/dev/video2`): Fixed overhead third-person camera providing global spatial context of the workspace, target container, and object initial pose.
  - `observation.images.wrist` (`/dev/video6`): Eye-in-hand wrist-mounted camera providing high-resolution local feedback during pre-grasp alignment and contact.

<div align="center">
  <img src="assets/hardware/camera_views_overview.png" width="98%" alt="Alicia-D Dual Camera Views Across Task Stages"/>
  <p><em>Figure 1: Synchronized Global Top Camera (top row) and Eye-in-Hand Wrist Camera (bottom row) observations captured at 30 Hz across four manipulation stages during a real-robot closed-loop rollout.</em></p>
</div>

### 1.2 Benchmark Dataset (`wipes-to-green-plate-01`)

| Property | Specification |
| :--- | :--- |
| **Task Instruction** | `"Pick up the pack of cleaning wipes and place it on the green plate"` |
| **Total Demonstrations** | **67 human teleoperated episodes** (`24,535` total synchronized frames) |
| **Sampling Frequency** | `30.0 Hz` (`13.6 minutes` of continuous high-precision demonstration data) |
| **Dataset Format** | LeRobot Dataset `v3.0` (`Parquet` state/action tables + `AV1 MP4` video chunks) |
| **Task Horizon & Phases** | ~360 frames (`12.0 s` average): (1) Reach & Pre-Grasp Alignment $\rightarrow$ (2) Deformable Wipes Grasp $\rightarrow$ (3) Lift & Cross-Table Transport $\rightarrow$ (4) Precision Placement into Green Plate |

---

## 2. Unified Cross-Model Adaptation Architecture

A core contribution of this repository is unifying **five structurally distinct policy families** under a **single hardware abstraction (`AliciaDFollower`)** and **standardized `30 Hz` closed-loop evaluation harness**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                Alicia-D 6-DoF Robot Hardware (30 Hz)                                     │
│   • Joint State: [joint1..6 (deg), gripper (0~1000)]  ───►  observation.state [7]                        │
│   • Dual Cameras: top (640×480), wrist (640×480)      ───►  observation.images.{top, wrist}              │
└────────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                     │
        ┌──────────────────────────┬─────────────────┴────────┬──────────────────────────┐
        ▼                          ▼                          ▼                          ▼
┌───────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐   ┌───────────────────────────┐
│ LeRobot Native    │    │ VLM Slot & Quantile │    │ StarVLA QwenGR00T    │   │ TurboVLA Alicia Adapter   │
│   (ACT 50k)       │    │ (SmolVLA / π0.5)    │    │ (Frozen Qwen3.5-2B)  │   │ (Frozen DINOv3 + MeanFlow)│
├───────────────────┤    ├─────────────────────┤    ├──────────────────────┤   ├───────────────────────────┤
│ • Direct top/wrist│    │ • Camera Slot Remap:│    │ • Dual-View [top,    │   │ • 7D -> 8D State Zero-Pad │
│ • Mean/Std Norm   │    │   camera1 / base_0  │    │   wrist] PIL 320×240 │   │   (Matches LIBERO 8D)     │
│ • 50-step CVAE    │    │ • Quantile [q01,q99]│    │ • q01/q99 Min-Max    │   │ • LIBERO Warm-Full Init   │
│ • Exp Temporal    │    │ • 10-step Flow ODE  │    │ • 10-step DiT-B Flow │   │ • 1-Step MeanFlow (1-NFE) │
│   Ensemble λ=0.01 │    │   + RTC Guidance    │    │   + Quintic C² Smooth│   │   + Quintic C² Smooth     │
└─────────┬─────────┘    └──────────┬──────────┘    └──────────┬───────────┘   └─────────────┬─────────────┘
          └─────────────────────────┴──────────────────────────┴─────────────────────────────┘
                                                     ▼
                             ┌───────────────────────────────────────────────┐
                             │   Unnormalized 7D Target Command [deg, 0~1000]│
                             │   Streamed to Alicia-D Controller @ 30.0 Hz   │
                             └───────────────────────────────────────────────┘
```

### Key Adaptation Mechanisms

1. **Camera Slot Standardization ([`policies/lerobot_adapters/camera_mapper.py`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/policies/lerobot_adapters/camera_mapper.py))**:
   - **ACT / TurboVLA / StarVLA**: Consume `observation.images.top` and `observation.images.wrist` directly (with `StarVLA` passing ordered `[top_pil, wrist_pil]` into `Qwen3.5-2B-VL`).
   - **SmolVLA**: Dynamically remaps `top -> observation.images.camera1` and `wrist -> observation.images.wrist` (`camera2`) to align with `SmolVLM2-500M`'s pretrained multi-view visual token slots.
   - **$\pi_{0.5}$**: Dynamically remaps `top -> observation.images.base_0_rgb` and `wrist -> observation.images.left_wrist_0_rgb` while zero-masking the unused right-wrist slot (`right_wrist_0_rgb`).
2. **Cross-Embodiment State Alignment & Selective Warm-Start ([`policies/turbovla_alicia/`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/policies/turbovla_alicia))**:
   - Pads the 7D Alicia-D state vector (`7D -> 8D`) to match `LIBERO` pretrained state encoders (`[J1..J6, Gripper, 0.0]`), linearly interpolates pretrained action queries (`12 -> 30`), and freezes the authentic real-world `DINOv3-ViT-B/16` (`--freeze_vision --keep_real_dinov3 --load_mode warm_full`).
3. **Full VLM Freezing for 2B+ Autoregressive Backbones ([`policies/starvla_alicia/`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/policies/starvla_alicia))**:
   - Freezes the entire `Qwen3.5-2B-VL` vision-language backbone (`freeze_module: "qwen_vl_interface"`, `FixQwen`) while training only the `209M` Flow-Matching `ActionModel (DiT-B)` and proprioceptive projector, eliminating single-scene language/vision overfitting and speeding up training by **`3.9×`**.
4. **Live Camera Viewpoint Alignment Tool ([`scripts/align_top_camera.py`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/align_top_camera.py))**:
   - Provides a real-time 50/50 alpha-blend + green Canny edge overlay + `ORB + RANSAC` pixel shift `(dx, dy)` HUD against training Episode 0 Frame 0 (`datasets/wipes-to-green-plate-01`), allowing sub-pixel (`<= 3 px`) physical re-alignment of the overhead camera before closed-loop evaluation.

---

## 3. Experimental Results & Cross-Model Benchmark Analysis

### 3.1 Trajectory Tracking Precision Across All 5 Policy Families

All five policy families (using each family's best configuration) were evaluated under an identical **15-step chunk horizon (`n_action_steps = 15`)** on the critical **5.0-second grasping and lifting window (Frames 150–300, 150 steps @ 30 Hz)** of `wipes-to-green-plate-01`, covering approach descent, gripper closure (`1000 -> 260`), and initial cross-table elevation.

<div align="center">
  <img src="assets/benchmarks/trajectory_tracking_all_models.png" width="100%" alt="5-Model Trajectory Tracking Comparison"/>
  <p><em>Figure 2: Cross-model multi-joint trajectory tracking comparison across TurboVLA (Warm-Full 12.5k), π0.5 (Full 20k), SmolVLA (30k), StarVLA (FixQwen 10k), and ACT (50k) against human ground-truth teleoperation across all 6 revolute joints and gripper opening.</em></p>
</div>

#### Per-Joint Mean Absolute Error Breakdown (Degrees — Best Variant per Policy Family)

| Policy Family (Best Variant) | Joint 1 (Base Yaw) | Joint 2 (Shoulder) | Joint 3 (Elbow) | Joint 4 (Forearm Roll) | Joint 5 (Wrist Pitch) | Joint 6 (Wrist Roll) | **6-Joint Mean MAE** $\downarrow$ | **6-Joint RMSE** $\downarrow$ | **Gripper MAE (0–1000)** $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACT (`50k`)** | 1.229° | 1.328° | 1.715° | 0.845° | 1.089° | 1.050° | 1.209° | 1.494° | 14.12 |
| **StarVLA (`FixQwen 10k`)** | 0.515° | 0.699° | 0.969° | 0.533° | 0.691° | 0.429° | 0.639° | 0.889° | 16.90 |
| **SmolVLA (`30k`)** | 0.366° | 0.437° | 0.636° | 0.309° | 0.302° | 0.268° | 0.386° | 0.501° | 26.43 |
| **$\pi_{0.5}$ (`Full 20k`)** | 0.245° | 0.555° | 0.562° | 0.278° | 0.361° | 0.292° | 0.382° | 0.518° | 19.23 |
| **TurboVLA (`Warm-Full 12.5k`)** | **0.118°** | **0.168°** | **0.216°** | **0.150°** | **0.170°** | **0.178°** | **0.166°** | **0.217°** | **6.53** |

---

### 3.2 Inference Latency & Computational Efficiency

We benchmarked end-to-end single-chunk inference latency (`bfloat16` mixed precision) across both edge/workstation deployment hardware (**NVIDIA RTX 3080 Laptop 16GB**) and server hardware (**NVIDIA RTX 4090 24GB**):

| Policy Family (Best Variant) | Total Params | Trainable Params | Denoising Steps (NFE) | Latency on RTX 3080 Laptop | Latency on RTX 4090 Server | Speedup vs. StarVLA / $\pi_{0.5}$ | Amortized Step Time ($K=15$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACT (`50k`)** | ~84M | 84M | **1 (CVAE Decoder)** | **8.8 ms** (113.6 Hz) | **6.1 ms** (163.9 Hz) | **18.9×** | 0.41 ms/step |
| **TurboVLA (`Warm-Full 12.5k`)** | ~216M | 20.6M | **1 (MeanFlow JVP)** | **24.6 ms** (40.7 Hz) | **16.8 ms** (59.5 Hz) | **6.8×** | 1.12 ms/step |
| **SmolVLA (`30k`)** | ~450M | 100M | 10 (Euler ODE) | 124.2 ms (8.1 Hz) | 56.3 ms (17.8 Hz) | 2.0× | 3.75 ms/step |
| **$\pi_{0.5}$ (`Full 20k`)** | ~3.3B | 3.3B | 10 (Euler ODE) | 294.7 ms (3.4 Hz) | 112.0 ms (8.9 Hz) | 1.0× | 7.47 ms/step |
| **StarVLA (`FixQwen 10k`)** | ~2.26B | 209M | 10 (Euler ODE) | 491.3 ms (2.0 Hz) | 115.0 ms (8.7 Hz) | 1.0× | 7.67 ms/step |

> **Key Takeaway**: Because `TurboVLA` distills flow velocity fields into a **single function evaluation (`1-NFE`)**, its full 30-step chunk inference takes only **`16.8 ms` on RTX 4090** (`24.6 ms` on RTX 3080 Laptop)—staying strictly below the **`33.3 ms` frame budget of a `30 Hz` control loop** while achieving the lowest tracking error (`0.166°`) across all five architectures.

---

### 3.3 Real-Robot Closed-Loop Temporal Smoothing Ablation Across All 5 Policy Families

Executing multi-step action chunks open-loop (`n_action_steps = 15..30`) without cross-chunk smoothing introduces sharp velocity discontinuities at chunk boundaries and high-frequency intra-chunk jitter. We evaluated **Unsmoothed (`No-Smooth`) vs. Smoothed (`Smooth`)** closed-loop execution on physical Alicia-D hardware across the **best configuration of all 5 policy families**:

1. **ACT (`Temporal Ensembling`, $\lambda = 0.01$)**: Re-queries the CVAE at every step (`n_action_steps=1`) and blends overlapping predictions via exponential temporal weighting $w_i = \exp(-\lambda \cdot i)$.
2. **SmolVLA & $\pi_{0.5}$ (`Real-Time Chunking / RTC`)**: Injects prefix guidance into the 10-step Flow-Matching Euler ODE solver (`--use_rtc=true`) to constrain newly denoised chunks to smoothly continue the previously executed trajectory prefix.
3. **StarVLA & TurboVLA (`RecedingHorizonTrajectoryBlender`, Quintic $C^2$ Anchor + Causal Gaussian Filter)**:
   - Applies intra-chunk causal Gaussian smoothing ($\sigma=1.8$), first-half anti-reversal clamping against `last_cmd`, **Quintic $C^2$ minimum-jerk boundary blending** ($s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$ over `K=12` steps), and monotonic gripper anti-bounce latching during transport.

<div align="center">
  <img src="assets/benchmarks/temporal_smoothing_ablation.png" width="100%" alt="5-Model Real-Robot Temporal Smoothing Ablation"/>
  <p><em>Figure 3: Real-robot closed-loop temporal smoothing ablation across all 5 policy families on physical Alicia-D hardware (top row: step-to-step action variation ||Δa_t||₂ across ACT, SmolVLA, π0.5, StarVLA, and TurboVLA; bottom row: mean step delta, 3rd-order joint jerk, velocity sign-flip rate, and maximum single-step jump).</em></p>
</div>

| Policy Family (Best Variant) | Execution Mode | Eval Run ID (`eval_runs/local/`) | Mean Step Delta $\|\Delta a_t\|_2$ (rad) $\downarrow$ | 95th Pct $\|\Delta a_t\|_2$ (rad) $\downarrow$ | Mean Accel (°/step²) $\downarrow$ | Mean Jerk (°/step³) $\downarrow$ | Velocity Sign-Flip Rate $\downarrow$ | Max Single-Step Jump (°) $\downarrow$ | Real-Robot Task Success |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACT (`50k`)** | Unsmoothed (`n=30`) | `eval_act_no_smooth` | 0.0109 | 0.0349 | 0.2765 | 0.5212 | 38.5% | 12.98° | 100% (1 Hz boundary jerks) |
| **ACT (`50k`)** | **Temporal Ensemble ($\lambda=0.01$)** | `eval_act_smooth` | **0.0053 (-51%)** | **0.0191 (-45%)** | **0.0244 (-91%)** | **0.0408 (-92%)** | **2.4%** | **2.45° (-81%)** | **100% (Ultra-smooth)** |
| **SmolVLA (`30k`)** | Unsmoothed (`n=15`) | `eval_smolvla_no_smooth` | 0.0111 | 0.0279 | 0.2541 | 0.4609 | 49.0% | 2.99° | 100% (Boundary pauses) |
| **SmolVLA (`30k`)** | **Flow-Matching RTC** | `eval_smolvla_smooth` | **0.0101 (-9%)** | **0.0268 (-4%)** | **0.2237 (-12%)** | **0.4108 (-11%)** | **47.1%** | 3.41° | **100% (Continuous flow)** |
| **$\pi_{0.5}$ (`Full 20k`)** | Unsmoothed (`n=15`) | `eval_pi05_no_smooth` | 0.0112 | 0.0262 | 0.1706 | 0.3163 | 24.9% | 7.71° | 100% (`7.71°` boundary kick) |
| **$\pi_{0.5}$ (`Full 20k`)** | **Flow-Matching RTC** | `eval_pi05_smooth` | **0.0106 (-5%)** | **0.0228 (-13%)** | **0.1357 (-20%)** | **0.2364 (-25%)** | 29.0% | **1.33° (-83%)** | **100% (No boundary kick)** |
| **StarVLA (`FixQwen 10k`)** | Unsmoothed (`n=15`) | `eval_starvla_fixqwen_no_smooth` | 0.0137 | 0.0260 | 0.3893 | 0.7096 | 54.9% | 3.15° | 100% (High-freq DiT jitter) |
| **StarVLA (`FixQwen 10k`)** | **Quintic $C^2$ Blender** | `eval_starvla_fixqwen_smooth` | **0.0060 (-56%)** | **0.0157 (-40%)** | **0.0212 (-95%)** | **0.0179 (-97%)** | **0.3%** | **0.88° (-72%)** | **100% (40× lower jerk)** |
| **TurboVLA (`Warm-Full 12.5k`)** | Unsmoothed (`n=15`) | `eval_turbovla_warm_full_no_smooth` | 0.0102 | 0.0326 | 0.2191 | 0.4067 | 35.8% | 5.98° | 100% (Sawtooth query jitter) |
| **TurboVLA (`Warm-Full 12.5k`)** | **Quintic $C^2$ Blender** | `eval_turbovla_warm_full_smooth` | **0.0052 (-49%)** | **0.0194 (-40%)** | **0.0224 (-90%)** | **0.0151 (-96%)** | **1.2%** | **1.31° (-78%)** | **100% (Lowest jerk: 0.0151)** |

---

### 3.4 Key Findings from Vertical Backbone Freezing & Transfer Ablations (`纵向消融核心结论`)

Detailed intra-model vertical ablations are provided in **[`starvla/README.md`](./starvla/README.md)** and **[`turbovla/README.md`](./turbovla/README.md)**:

<div align="center">
  <img src="assets/benchmarks/turbovla_warmstart_vs_scratch.png" width="100%" alt="TurboVLA Cross-Embodiment Transfer & Selective Warm-Start Ablation"/>
  <p><em>Figure 4: TurboVLA 5-way cross-embodiment transfer & selective warm-start ablation (see <a href="./turbovla/README.md">turbovla/README.md</a> for full details) and StarVLA 3-way VLM freezing ablation (see <a href="./starvla/README.md">starvla/README.md</a>).</em></p>
</div>

1. **The Open-Loop Memorization Trap on Single-Scene Real-Robot Datasets**:
   - In both `TurboVLA` (`Unfrozen Warm-Start 20k`, `0.088°` open-loop MAE) and `StarVLA` (`Unfrozen Qwen3.5-2B-VL 20k`, `0.354°` open-loop MAE), unfreezing large vision or VLM backbones on 67 demonstrations causes the model to memorize training background lighting cues and fail (`0%` real-robot success) under live closed-loop deployment.
2. **Full Pretrained Backbone Freezing (`FixQwen` & `Frozen Real DINOv3`) Enables 100% Real-Robot Generalization**:
   - Freezing the entire `Qwen3.5-2B-VL` backbone in `StarVLA` (`FixQwen 10k`) reduces trainable parameters from `2.26B` to `209M`, cuts training time by **`3.9×` (`171 min -> 44 min`)**, and achieves **100% real-robot closed-loop success**.
   - Freezing the authentic real-world `DINOv3-ViT-B/16` (`LVD-1689M`) while warm-starting the transformer enhancer and 1-NFE MeanFlow head from `LIBERO 40k` (`TurboVLA Warm-Full 12.5k`) cuts training time by **`4.5×` (`163 min -> 36 min`)**, improves open-loop MAE from `0.606°` (`8k`) to **`0.166°` (`12.5k`)**, and achieves **100% real-robot closed-loop success** even without smoothing.

---

## 4. Repository Structure

```text
alicia-d-vla-benchmark/
├── README.md                                  # Main cross-model benchmark report ('横向对比 / 总比')
├── act/                                       # [ACT Vertical Deep-Dive ('纵向分比')]
│   ├── README.md                              # ACT dedicated ablation report
│   ├── act_smooth_vs_no_smooth.png            # 6-joint trajectory & smoothness comparison
│   ├── rollout_comparison_act.gif             # Side-by-side No-Smooth vs. Temporal Ensemble GIF
│   ├── train_act_4090.sh
│   └── eval_act_{smooth,no_smooth}.sh
├── smolvla/                                   # [SmolVLA Vertical Deep-Dive ('纵向分比')]
│   ├── README.md                              # SmolVLA dedicated ablation report
│   ├── smolvla_smooth_vs_no_smooth.png        # 6-joint trajectory & RTC smoothness comparison
│   ├── smolvla_horizon_ablation.png           # Execution horizon (n=15, 20, 30) ablation plot
│   ├── rollout_comparison_smolvla.gif         # Side-by-side No-Smooth vs. Flow-Matching RTC GIF
│   ├── train_smolvla_4090.sh
│   └── eval_smolvla_{smooth,no_smooth}.sh
├── pi0_5/                                     # [π0.5 Vertical Deep-Dive ('纵向分比')]
│   ├── README.md                              # π0.5 dedicated ablation report
│   ├── pi05_smooth_vs_no_smooth.png           # 6-joint trajectory & boundary jump suppression
│   ├── rollout_comparison_pi05.gif            # Side-by-side No-Smooth vs. Flow-Matching RTC GIF
│   ├── train_pi05_{a800_full,4090_lora}.sh
│   ├── merge_pi05_lora.py
│   └── eval_pi05_{smooth,no_smooth}.sh
├── starvla/                                   # [StarVLA Vertical Deep-Dive ('纵向分比')]
│   ├── README.md                              # StarVLA 3-way VLM freeze & Quintic C² report
│   ├── starvla_3way_freeze_ablation.png       # Unfrozen vs. FreezeViT vs. FixQwen ablation
│   ├── starvla_smooth_vs_no_smooth.png        # 6-joint trajectory & Quintic C² blender plot
│   ├── rollout_comparison_starvla.gif         # Side-by-side No-Smooth vs. Quintic C² Smooth GIF
│   ├── run_wipes_{fixqwen,freeze_vit}_4090.sh
│   ├── starvla_wipes_qwen3_5_2b_gr00t_{fixqwen,freeze_vit}.yaml
│   └── eval_starvla_{smooth,no_smooth}.sh
├── turbovla/                                  # [TurboVLA Vertical Deep-Dive ('纵向分比')]
│   ├── README.md                              # TurboVLA 5-way transfer & Quintic C² report
│   ├── turbovla_3way_transfer_ablation.png    # Scratch vs. Sim-ViT vs. Frozen-ViT vs. Warm-Body/Full
│   ├── turbovla_smooth_vs_no_smooth.png       # 6-joint trajectory & Quintic C² blender plot
│   ├── rollout_comparison_turbovla.gif        # Side-by-side No-Smooth vs. Quintic C² Smooth GIF
│   ├── train_turbovla_wipes_{selective,warm_ab,freeze_vit}.sh
│   └── eval_turbovla_warm_{full,body}_{smooth,no_smooth}.sh
├── assets/                                    # Cross-model ('横向对比 / 总比') figures & rollout GIFs
│   ├── demos/                                 # 5 Real-robot closed-loop rollout GIFs (Best per policy)
│   │   ├── rollout_turbovla_smooth.gif        # TurboVLA (Warm-Full 12.5k, Quintic C² Smooth)
│   │   ├── rollout_starvla_smooth.gif         # StarVLA (FixQwen 10k, Quintic C² Smooth)
│   │   ├── rollout_pi05_rtc.gif               # π0.5 (Full 20k, Flow-Matching RTC)
│   │   ├── rollout_smolvla_ensemble.gif       # SmolVLA (30k, Flow-Matching RTC)
│   │   └── rollout_act50_ensemble.gif         # ACT (50k, Temporal Ensembling λ=0.01)
│   ├── hardware/
│   │   └── camera_views_overview.png          # Synchronized dual-camera stage overview
│   └── benchmarks/
│       ├── trajectory_tracking_all_models.png # 5-model joint & gripper tracking curves
│       ├── turbovla_warmstart_vs_scratch.png  # Cross-Embodiment & Selective Warm-Start ablation
│       └── temporal_smoothing_ablation.png    # 5-model real-robot temporal smoothing ablation
├── hardware/                                  # LeRobot hardware plugin for Alicia-D 6-DoF manipulator
│   ├── alicia_d_follower.py                   # Serial driver, kinematic state reader & 30 Hz action streamer
│   └── config_alicia_d_follower.py            # Dataclass configuration & camera registration
├── policies/
│   ├── lerobot_adapters/                      # Unified adapters & training scripts for ACT, SmolVLA, π0.5
│   ├── starvla_alicia/                        # StarVLA Qwen3.5-2B-VL + GR00T DiT-B training configs & scripts
│   └── turbovla_alicia/                       # TurboVLA cross-embodiment adaptation package
├── scripts/
│   ├── align_top_camera.py                    # Live overhead camera overlay & ORB pixel-shift alignment tool
│   ├── eval_alicia_arms.py                    # Real-robot closed-loop evaluator for ACT / SmolVLA / π0.5
│   ├── eval_alicia_starvla.py                 # Real-robot closed-loop evaluator for StarVLA + Quintic C² blender
│   ├── eval_alicia_turbovla.py                # Real-robot closed-loop evaluator for TurboVLA + Quintic C² blender
│   ├── generate_repo_visuals.py               # Reproducible cross-model ('总比') figure & GIF generator
│   └── generate_per_model_assets.py           # Reproducible per-model ('分比') figure & GIF generator
└── docs/
    ├── technical_mapping.md                   # Deep-dive mathematical & architectural mapping spec
    ├── train_starvla_{fixqwen,freeze_vit,unfrozen}.log
    └── train_wipes_{warm_full,warm_body,freeze_vit,a_warmstart,b_scratch}.log
```

---

## 5. Quick Start & Reproducible Workflows

### 5.1 Environment Installation

```bash
conda create -n lerobot python=3.11 -y
conda activate lerobot

# Install LeRobot v0.4.4 + Alicia-D SDK & PyAV video decoder
pip install -e ./lerobot
pip install av pyarrow opencv-python matplotlib pillow
```

### 5.2 Training All 5 Policy Families

```bash
# 1. TurboVLA (Frozen Real-World DINOv3 + LIBERO 40k Warm-Full, 12.5k Steps — 36 min on RTX 4090)
bash scripts/train_turbovla_wipes_warm_ab.sh

# 2. StarVLA (Frozen Qwen3.5-2B-VL + Flow-Matching DiT-B, 10k Steps — 44 min on RTX 4090)
bash policies/starvla_alicia/run_wipes_fixqwen_4090.sh

# 3. ACT (50,000 steps on RTX 4090)
bash policies/lerobot_adapters/train_act_4090.sh

# 4. SmolVLA (30,000 steps on RTX 4090)
bash policies/lerobot_adapters/train_smolvla_4090.sh

# 5. π0.5 (Full Fine-Tuning on A800 80GB or LoRA on RTX 4090 24GB)
bash policies/lerobot_adapters/train_pi05_a800_full.sh
```

### 5.3 Real-Robot Closed-Loop Evaluation (30 Hz on Alicia-D)

#### Step 1: Align Overhead Camera Viewpoint (`<= 3 px` shift vs. Training Episode 0)
```bash
conda activate lerobot
python scripts/align_top_camera.py
```

#### Step 2: Run Closed-Loop Evaluation (Best Variant of Each Policy — Smoothed vs. Unsmoothed)
```bash
# 1. TurboVLA (Warm-Full 12.5k) — Quintic C² Smoothed vs. Unsmoothed
bash scripts/eval_turbovla_warm_full_smooth.sh
bash scripts/eval_turbovla_warm_full_no_smooth.sh

# 2. StarVLA (FixQwen 10k) — Quintic C² Smoothed vs. Unsmoothed
bash scripts/eval_starvla_smooth.sh
bash scripts/eval_starvla_no_smooth.sh

# 3. π0.5 (Full 20k) — Flow-Matching RTC Smoothed vs. Unsmoothed
bash scripts/eval_pi05_smooth.sh
bash scripts/eval_pi05_no_smooth.sh

# 4. SmolVLA (30k) — Flow-Matching RTC Smoothed vs. Unsmoothed
bash scripts/eval_smolvla_smooth.sh
bash scripts/eval_smolvla_no_smooth.sh

# 5. ACT (50k) — Exponential Temporal Ensembling (λ=0.01) vs. Unsmoothed
bash scripts/eval_act_smooth.sh
bash scripts/eval_act_no_smooth.sh
```

### 5.4 Reproduce All Cross-Model & Per-Model Plots and GIFs

```bash
python scripts/generate_repo_visuals.py
python scripts/generate_per_model_assets.py
```

---

## 6. Citation & Acknowledgments

This benchmark builds upon [HuggingFace LeRobot](https://github.com/huggingface/lerobot), [Physical Intelligence $\pi_{0.5}$](https://www.physicalintelligence.company/), [SmolVLA](https://huggingface.co/blog/smolvla), [StarVLA](https://github.com/StarVLA/StarVLA), [ACT](https://tonyzhaozh.github.io/aloha/), and [TurboVLA](https://arxiv.org/).
