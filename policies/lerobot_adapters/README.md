# LeRobot Policy Adapters & Training Recipes

本目录包含 **Alicia-D** 机械臂接入 **LeRobot** 框架的全套训练启动脚本、PEFT LoRA 合并工具、观测命名空间映射适配器与真机时序平滑处理模块。

---

## 🗂️ 目录文件结构

```text
policies/lerobot_adapters/
├── README.md                      # 本说明文档
├── __init__.py                    # 模块导出定义
├── merge_pi05_lora.py             # π0.5 (4.14B) PEFT LoRA 权重合入基座与预处理文件打包工具
├── temporal_ensembler.py          # ACTTemporalEnsembler 独立时序指数加权平滑器
├── rtc_processor.py               # π0.5 实时分块 (RTC) 前缀引导与连续性处理器
├── camera_mapper.py               # 跨模型视觉观测键名自适应映射 (base_0_rgb / camera1 / top)
├── train_pi05_a800_full.sh        # π0.5 (4.14B) A800 单卡 30k 全参数微调生产启动脚本
├── train_pi05_4090_lora.sh        # π0.5 (4.14B) RTX 4090 单卡 LoRA (rank=16) 微调脚本
├── train_smolvla_4090.sh          # SmolVLA (0.45B) RTX 4090 30k (batch_size=64) 训练脚本
├── train_act_4090.sh              # ACT (80M) RTX 4090 50k (chunk=100) 从头训练脚本
└── configs/                       # 各模型对应的原始训练参数配置文件
    ├── train_config_pi05_full.json
    ├── train_config_pi05_lora.json
    ├── train_config_smolvla.json
    └── train_config_act.json
```

---

## 🚀 训练与微调指南 (Training Recipes)

### 1. $\pi_{0.5}$ (4.14B) 全量微调 (Full Fine-Tuning on A800)
连同 PaliGemma-3B 视觉骨干与 300M Action Expert 一同反向传播更新：
```bash
bash policies/lerobot_adapters/train_pi05_a800_full.sh
```

### 2. $\pi_{0.5}$ (4.14B) 轻量化 LoRA 微调 (RTX 4090 24GB)
仅对自注意力与交叉注意力投影层微调（`rank=16`, `alpha=32`，显存仅占约 18GB）：
```bash
bash policies/lerobot_adapters/train_pi05_4090_lora.sh
# 训练完成后，一键合并 LoRA 权重输出独立 LeRobot Checkpoint:
python policies/lerobot_adapters/merge_pi05_lora.py \
  --base_model models/pi05_base \
  --lora_dir outputs/pi05_wipes_4090_lora_ck30_bs3_30000/030000/pretrained_model \
  --output_dir weights/pi05_wipes_lora_merged
```

### 3. SmolVLA (0.45B) 训练
```bash
bash policies/lerobot_adapters/train_smolvla_4090.sh
```

### 4. ACT (80M) 训练
```bash
bash policies/lerobot_adapters/train_act_4090.sh
```

---

## 🔧 核心组件接口说明

1. **`ACTTemporalEnsembler`**：
   在每个控制步获取最新生成的动作块，按 $w_i = \exp(-m \cdot i)$ 指数衰减加权历史重叠步，解决 chunk 边缘机械臂速度跳跃与顿挫。
2. **`RTCProcessor`**：
   专为 Flow Matching 策略设计，在执行 `n_action_steps` 触发重规划时，将旧动作块未执行的剩余步作为前缀注入 ODE 去噪采样，强制轨迹一阶连续。
3. **`CameraMapper`**：
   统一真实物理相机的 `top / wrist` 键名与各算法预训练期望特征名，消除多模型评测时的样板代码。
