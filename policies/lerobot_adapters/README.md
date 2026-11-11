# LeRobot Policy Adapters & Configs

本目录包含 **Alicia-D** 机械臂接入 **LeRobot** 框架的 PEFT LoRA 合并工具、观测命名空间映射适配器、动作时序平滑处理模块及各模型超参数配置文件。

---

## 🗂️ 目录文件结构

```text
policies/lerobot_adapters/
├── README.md                      # 本说明文档
├── __init__.py                    # 模块导出定义
├── merge_pi05_lora.py             # π0.5 (~3.3B) PEFT LoRA 权重合入基座与预处理文件打包工具
├── temporal_ensembler.py          # ACTTemporalEnsembler 独立时序指数加权平滑器
├── rtc_processor.py               # π0.5 实时分块 (RTC) 前缀引导与连续性处理器
├── camera_mapper.py               # 跨模型视觉观测键名自适应映射 (base_0_rgb / camera1 / top)
└── configs/                       # 各模型对应的训练超参数配置文件
    ├── train_config_pi05_full.json
    ├── train_config_pi05_lora.json
    ├── train_config_smolvla.json
    └── train_config_act.json
```

---

## 🚀 模型权重合并与配置说明 (Adapter & Merge Utilities)

### 1. $\pi_{0.5}$ (~3.3B) LoRA 权重合入基座
对基于 `rank=16, alpha=32`（开启 `gradient_checkpointing=true` 后实测峰值显存 `9.75 GB`）微调得到的 LoRA Checkpoint，一键合入基座并打包归一化统计量：
```bash
python policies/lerobot_adapters/merge_pi05_lora.py \
  --base_model models/pi05_base \
  --lora_dir outputs/pi05_wipes_4090_lora_ck30_bs3_30000/030000/pretrained_model \
  --output_dir weights/pi05_wipes_lora_merged
```

### 2. 各策略训练超参数归档 (`configs/`)
- `configs/train_config_pi05_full.json`：$\pi_{0.5}$ A800 全参数微调配置
- `configs/train_config_pi05_lora.json`：$\pi_{0.5}$ RTX 4090 LoRA (`rank=16`) 微调配置
- `configs/train_config_smolvla.json`：SmolVLA (0.45B) 30k 微调配置
- `configs/train_config_act.json`：ACT (`chunk_size=100`, `batch_size=4`, 50k) 从头训练配置

---

## 🔧 核心组件接口说明

1. **`ACTTemporalEnsembler`**：
   在每个推理步获取最新生成的动作块，按 $w_i = \exp(-m \cdot i)$ 指数衰减加权历史重叠预测，消除 Action Chunk 边界的不连续跳变。
2. **`RTCProcessor`**：
   专为 Flow Matching 策略设计，在执行 `n_action_steps` 触发重规划时，将旧动作块未执行的剩余步作为前缀注入 ODE 去噪采样，约束采样流场在 Action Chunk 边界保持时序连续性。
3. **`CameraMapper`**：
   统一真实物理相机的 `top / wrist` 键名与各算法预训练期望特征名，消除多模型评测时的样板代码。
