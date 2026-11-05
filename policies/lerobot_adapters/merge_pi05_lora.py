#!/usr/bin/env python3
"""
Merge PEFT LoRA adapters into the Pi0.5 (4.14B) base checkpoint and bundle
LeRobot preprocessor/postprocessor normalizer statistics for standalone real-robot inference.
"""

import argparse
import os
import shutil
import torch
from peft import PeftModel
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig


def main():
    parser = argparse.ArgumentParser(description="Merge Pi0.5 LoRA adapter into standalone LeRobot checkpoint")
    parser.add_argument("--base_model", type=str, default="/home/yunyi/Desktop/vla/models/pi05_base",
                        help="Path to base Pi0.5 pretrained model")
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="Path to LoRA fine-tuned checkpoint directory (e.g. step 30000/pretrained_model)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged standalone model")
    args = parser.parse_args()

    print("=== merge start ===")
    cfg = PreTrainedConfig.from_pretrained(args.lora_dir)
    print(f"config loaded, type = {cfg.type}")

    # Load base policy
    cfg.pretrained_path = args.base_model
    base_policy = PI05Policy.from_pretrained(args.base_model, config=cfg)
    total_params = sum(p.numel() for p in base_policy.parameters()) / 1e9
    print(f"base loaded, params {total_params:.2f} B")

    # Attach & merge LoRA adapter
    peft_model = PeftModel.from_pretrained(base_policy, args.lora_dir)
    print("adapter attached")
    merged_policy = peft_model.merge_and_unload()
    print(f"merged OK, params {total_params:.2f} B")

    os.makedirs(args.output_dir, exist_ok=True)
    merged_policy.save_pretrained(args.output_dir)
    print(f"model saved to {args.output_dir}")

    # Copy LeRobot preprocessor/postprocessor normalizer & training config files
    aux_files = [
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]
    for fname in aux_files:
        src = os.path.join(args.lora_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            print(f"copied {fname}")

    print("=== merge done ===")


if __name__ == "__main__":
    main()
