# StarVLA (Qwen3.5-2B + QwenGR00T DiT-B Flow-Matching) 湿巾抓取任务交接文档

**生成时间**：2026-11-05 17:06  
**任务名称**：`Pick up the pack of cleaning wipes and place it on the green plate`（单臂 6-DoF + 夹爪，`action_dim=7, action_horizon=30, fps=30, image_size=224x224`）

---

## 一、刚刚完成的真机评测（16:45 无平滑 & 16:50 有平滑）复盘与根因诊断

刚刚使用 `starvla_wipes_qwen3_5_2b_gr00t_fixqwen`（全冻结 `Qwen3.5-2B` VLM 主干、仅训练 `DiT-B` 动作头 10,000 步）进行了两轮真机评测：
- 无平滑记录：[`/media/yunyi/Data/Desktop/work/vla/eval_runs/local/eval_starvla_no_smooth`](file:///media/yunyi/Data/Desktop/work/vla/eval_runs/local/eval_starvla_no_smooth)（511 帧）
- 有平滑记录：[`/media/yunyi/Data/Desktop/work/vla/eval_runs/local/eval_starvla_smooth`](file:///media/yunyi/Data/Desktop/work/vla/eval_runs/local/eval_starvla_smooth)（218 帧）

### 1. 现象与三大根因
1. **根因一：`fixqwen` 全冻结通用 `Qwen3.5-2B`（ViT + LLM 全冻结）导致空间坐标退化为数据集均值（`J1 = +5.8°` 偏左抓空）**
   - 在 [`QwenGR00T.py`](file:///media/yunyi/Data/Desktop/work/vla/starVLA/starVLA/model/framework/VLM4A/QwenGR00T.py#L183-L190) 中，`DiT-B` 动作头仅通过 Cross-Attention 读取 `Qwen3.5-2B` 语言模型**最后一层**的 `hidden_states[-1]`，中间没有任何独立的二维空间视觉 Adapter。
   - 原版 `Qwen3.5-2B` 是通用图文对话模型（未经机器人轨迹预训练），其最后一层隐藏状态针对下一个文本 Token 预测优化，而非精细二维工作空间坐标。当 `freeze_modules: "qwen_vl_interface"` 将 ViT 和 LLM 全部冻结时，`DiT-B` 无法调整 LLM 顶层表征以区分湿巾精确位置，导致模型在下降阶段（`Step 0..90`）直接向数据集全局均值（`mean J1 = +7.25°`）坍缩，停在湿巾与绿盘中间的死区（`J1 = +5.8°, J2 = -33.2°, J3 = +44.1°`，位于湿巾左侧约 2.5 cm）悬停至 `Step 300` 后抓空，随后才执行抬升并转向绿盘（`J1 = +13.4°, J2 = -40.5°`）张开夹爪。
2. **根因二：[`eval_alicia_starvla.py`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_alicia_starvla.py) 缺少相机自动曝光预热（首帧惨绿失真）**
   - 对比 `eval_alicia_turbovla.py`（连接后预读 15 帧相机画面完成自动曝光与白平衡收敛），旧版 `eval_alicia_starvla.py` 在 `robot.connect()` 后立即进入 `record_loop` 推理第 0 帧。
   - 真机录制的 `frame-000000.png` 显示腕部相机首帧严重欠曝且呈深绿色偏色，导致第 0 个 30 步 Chunk 从起步瞬间就将 `J1` 从 `+1.3°` 直接拉偏至 `+4.3°`。
3. **根因三：`DiT-B` Flow-Matching 采样噪声导致 Chunk 内部高频锯齿抖动（`SignFlip = 59.1%`），且旧版平滑器仅平滑前 5 步**
   - 在 [`GR00T_ActionHeader.py`](file:///media/yunyi/Data/Desktop/work/vla/starVLA/starVLA/model/modules/action_model/GR00T_ActionHeader.py#L382-L428) 中，`predict_action` 每次从全序列独立高斯白噪声 `torch.randn(B, 30, 7)` 出发做 10 步 Euler 积分，无时域卷积约束，导致单次预测的 30 步 Chunk 内部相邻步速度符号翻转率高达 **`59.1%`**（`MeanJerk = 0.789°`）。
   - 旧版 `eval_alicia_starvla.py` 在 `n_action_steps = 30` 时，仅将前 5 步与上一 Chunk 末尾做指数加权（权重从 `0.95` 降到 `0.78` 后在第 5 步突降为 `0`），后 25 步完全未做平滑，不仅保留了 `51.7%` 的高频翻转抖动，还在第 5 步制造了人为阶跃跳变。

---

## 二、已完成的修复与服务器 `freeze_vit` 训练进展

### 1. 评测代码 [`eval_alicia_starvla.py`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_alicia_starvla.py) 已修复
1. **新增 15 帧相机自动曝光与白平衡预热**（[`eval_alicia_starvla.py`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_alicia_starvla.py#L443-L447)）：连接机械臂后先连续读取 15 帧观测，彻底消除首帧深绿偏色。
2. **接入全时域因果平滑器 [`RecedingHorizonTrajectoryBlender`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_alicia_starvla.py#L159-L246)**：
   - 保持 `replan_steps = n_action_steps = 30` 全 Chunk 执行，不截断后半段闭合与抬升动作；
   - 对 6 个机械臂关节施加 Chunk 内反射填充一维高斯平滑（$\sigma=1.8$），滤除 `DiT-B` 高斯采样引起的 `59.1%` 锯齿震荡；
   - 施加前半段防回撤单调钳位 + 五次多项式最小加加速度交界锚定过渡（`anchor_steps=12`）+ 抓取搬运期夹爪单调防回弹锁存。

### 2. 三版 StarVLA 训练权重对比与 `4090` 实时状态

| 模型版本 | 冻结策略 (`freeze_modules`) | 训练步数 / 状态 | 核心特点与诊断 |
| :--- | :--- | :--- | :--- |
| `wipes_qwen3_5_2b_gr00t_retrain` | 无冻结（全量微调 ViT + LLM + DiT） | 20,000 步（已完成，约 26 Epochs） | 视觉塔 `visual` 被完全放开微调 26 个 Epoch，在 81 条单任务数据上严重过拟合背景像素。 |
| `wipes_qwen3_5_2b_gr00t_fixqwen` | 冻结整个 VLM（`qwen_vl_interface`） | 10,000 步（已完成，约 13 Epochs） | LLM 最后一层文本隐藏状态未适配空间坐标回归，`J1` 向数据集均值（`+5.8°`）坍缩偏左。 |
| **`wipes_qwen3_5_2b_gr00t_freeze_vit`（推荐首选）** | **仅冻结视觉塔（`qwen_vl_interface.model.model.visual`），微调 LLM (`lr=1e-5`) + DiT-B (`lr=1e-4`)** | **`steps_8000` 已于 16:26 保存就绪**；`steps_10000` 正在 `4090` 冲刺（当前 `8550/10000`，Loss 已降至 `0.0045`） | **兼顾视觉防过拟合（冻结预训练 ViT）与空间坐标精确对齐（微调 LLM 跨模态层 + DiT）！** |

---

## 三、新会话操作指南

### 1. 拉取并评测 `freeze_vit` 8,000 步（或 10,000 步）权重
- **查看 `4090` 上 `freeze_vit` 训练是否到达 10,000 步**：
  ```bash
  ssh -n 4090 "ls -lh /home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_qwen3_5_2b_gr00t_freeze_vit/checkpoints/ && tail -n 5 /home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_freeze_vit.log"
  ```
- **拉取 `freeze_vit` 权重到本地**（当前 `steps_8000_pytorch_model.pt` 已随时可拉取；约 18:50 后可拉取 `steps_10000_pytorch_model.pt`）：
  ```bash
  mkdir -p /media/yunyi/Data/Desktop/work/vla/weights/starvla_wipes_qwen3_5_2b_gr00t_freeze_vit
  scp 4090:/home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_qwen3_5_2b_gr00t_freeze_vit/config.yaml \
      4090:/home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_qwen3_5_2b_gr00t_freeze_vit/dataset_statistics.json \
      /media/yunyi/Data/Desktop/work/vla/weights/starvla_wipes_qwen3_5_2b_gr00t_freeze_vit/
  scp 4090:/home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_qwen3_5_2b_gr00t_freeze_vit/checkpoints/steps_8000_pytorch_model.pt \
      /media/yunyi/Data/Desktop/work/vla/weights/starvla_wipes_qwen3_5_2b_gr00t_freeze_vit/pytorch_model.pt
  ```

### 2. 启动真机评测脚本
将 [`eval_starvla_smooth.sh`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_starvla_smooth.sh) 与 [`eval_starvla_no_smooth.sh`](file:///media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_starvla_no_smooth.sh) 中的 `--ckpt_path` 指向 `starvla_wipes_qwen3_5_2b_gr00t_freeze_vit/pytorch_model.pt` 后运行：
```bash
# 1. 有平滑评测（推荐首选，包含 Chunk 内高斯滤波 + 五次多项式 C^2 锚定过渡 + 15 帧相机预热）
bash /media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_starvla_smooth.sh

# 2. 无平滑基线对比
bash /media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts/eval_starvla_no_smooth.sh
```
