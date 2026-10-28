#!/usr/bin/env python3
"""TurboVLA fine-tuning on real robot Alicia-D wipes dataset."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from turbovla.models.turbovla import build_turbovla
from experiments.wipes.wipes_dataset import LeRobotWipesDataset, wipes_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TurboVLA on Alicia-D wipes dataset.")
    parser.add_argument("--dataset_dir", type=str, default="/home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01")
    parser.add_argument("--stats_path", type=str, default="/home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json")
    parser.add_argument("--dinov3_path", type=str, default="/home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--bert_path", type=str, default="/home/yunyi/Desktop/vla/models/bert-base-uncased")
    parser.add_argument("--pretrained_ckpt", type=str, default="/home/yunyi/Desktop/vla/TurboVLA/outputs/libero_g128_40k/turbovla_step_40000.pth")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/turbovla_wipes_ck30")
    parser.add_argument("--checkpoint_prefix", type=str, default="turbovla_wipes_step")

    # Architecture
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_state_tokens", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--enhancer_inner_dim", type=int, default=1024)
    parser.add_argument("--expected_image_size", type=int, default=256)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--train_text_encoder", dest="freeze_text_encoder", action="store_false")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--dinov3_lr", type=float, default=2.5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.025)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="turbovla-wipes")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.025):
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (float(step + 1) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name: str, param: torch.Tensor) -> bool:
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias") or "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def load_and_interpolate_checkpoint(model: nn.Module, ckpt_path: str, chunk_size: int):
    print(f"[Init] Loading pretrained checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))

    cleaned_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        cleaned_sd[k] = v

    q_key = "action_head.decoder.action_queries.weight"
    if q_key in cleaned_sd and cleaned_sd[q_key].shape[0] != chunk_size:
        old_q = cleaned_sd[q_key]
        new_q = F.interpolate(old_q.T.unsqueeze(0), size=chunk_size, mode="linear", align_corners=True).squeeze(0).T
        cleaned_sd[q_key] = new_q
        print(f"[Init] Linearly interpolated {q_key}: {old_q.shape} -> {new_q.shape}")

    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    print(f"[Init] Pretrained checkpoint loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 80)
    print("TurboVLA Fine-tuning on Alicia-D Wipes Dataset")
    print(f"Device: {device}, Chunk Size: {args.chunk_size}, Batch Size: {args.batch_size} (accum: {args.grad_accum_steps})")
    print(f"Steps: {args.max_steps}, Head LR: {args.head_lr}, DINO LR: {args.dinov3_lr}, EMA: {args.ema_decay}")
    print("=" * 80)

    class ModelArgs:
        dinov3_path = args.dinov3_path
        bert_path = args.bert_path
        hidden_dim = args.hidden_dim
        nheads = args.nheads
        dim_feedforward = args.dim_feedforward
        max_text_len = 256
        text_padding_length = 21
        text_padding_length_by_instruction = {}
        vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
        enhancer_inner_dim = args.enhancer_inner_dim
        text_dropout = 0.0
        fusion_dropout = 0.0
        fusion_droppath = 0.1
        action_dim = args.action_dim
        chunk_size = args.chunk_size
        state_dim = args.state_dim
        num_state_tokens = args.num_state_tokens
        local_files_only = True
        freeze_vision_encoder = args.freeze_vision
        freeze_text_encoder = args.freeze_text_encoder
        dinov3_precision = "bf16_autocast"
        num_views = 2
        image_size = args.expected_image_size
        position_embedding = "view"
        encode_views_separately = True
        padding_strategy = "key_padding_mask"

    model = build_turbovla(ModelArgs())

    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        load_and_interpolate_checkpoint(model, args.pretrained_ckpt, args.chunk_size)
    else:
        print(f"[Warning] Pretrained checkpoint not found at {args.pretrained_ckpt}. Training from scratch.")

    if args.freeze_vision:
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder.backbone"):
                param.requires_grad = False
        print("[Frozen] Vision backbone (DINOv3) frozen.")

    model.to(device)

    dataset = LeRobotWipesDataset(
        dataset_dir=args.dataset_dir,
        stats_path=args.stats_path,
        chunk_size=args.chunk_size,
        expected_image_size=args.expected_image_size,
        split="train",
        split_ratio=args.split_ratio,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=wipes_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    grouped = {
        ("head_decay", args.head_lr, args.weight_decay): [],
        ("head_no_decay", args.head_lr, 0.0): [],
        ("dinov3_decay", args.dinov3_lr, args.weight_decay): [],
        ("dinov3_no_decay", args.dinov3_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_dino = name.startswith("vision_encoder.backbone")
        decay = _use_weight_decay(name, param)
        if is_dino:
            key = ("dinov3_decay" if decay else "dinov3_no_decay", args.dinov3_lr, args.weight_decay if decay else 0.0)
        else:
            key = ("head_decay" if decay else "head_no_decay", args.head_lr, args.weight_decay if decay else 0.0)
        grouped[key].append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd, "name": gname}
        for (gname, lr, wd), params in grouped.items()
        if params
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scheduler = build_scheduler(optimizer, args.max_steps, args.warmup_steps, args.min_lr_ratio)

    # Track EMA weights for smooth real-robot inference
    ema_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.is_floating_point()
    }

    wandb_run = None
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run_name or os.path.basename(args.checkpoint_dir)
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"[WandB] Active run: {wandb_run.url}")

    model.train()
    global_step = 0
    accum_loss = 0.0
    start_time = time.time()
    data_iter = iter(dataloader)

    print(f"\n[Start] Beginning fine-tuning for {args.max_steps} steps on {len(dataset)} frames...\n")

    while global_step < args.max_steps:
        optimizer.zero_grad()
        loss_window = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            samples, instructions, states, target_actions, action_masks = batch

            samples = {k: v.to(device, non_blocking=True) for k, v in samples.items()}
            states = states.to(device, non_blocking=True)
            target_actions = target_actions.to(device, non_blocking=True)
            action_masks = action_masks.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_actions = model(instructions, samples, states)
                loss = masked_l1_loss(pred_actions, target_actions, action_masks)

            sub_loss = loss / args.grad_accum_steps
            sub_loss.backward()
            loss_window += loss.item() / args.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_params:
                    ema_params[name].mul_(args.ema_decay).add_(param.detach(), alpha=1.0 - args.ema_decay)

        global_step += 1
        accum_loss += loss_window

        if global_step % args.log_freq == 0:
            avg_loss = accum_loss / args.log_freq
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(1e-5, elapsed)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Step {global_step:05d}/{args.max_steps:05d} | "
                f"Loss: {avg_loss:.4f} | "
                f"HeadLR: {current_lr:.2e} | "
                f"GradNorm: {float(grad_norm):.2f} | "
                f"Speed: {steps_per_sec:.2f} step/s"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/grad_norm": float(grad_norm),
                    },
                    step=global_step,
                )
            accum_loss = 0.0

        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step:05d}.pth")
            raw_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            ema_sd = dict(raw_sd)
            for name, ema_t in ema_params.items():
                if name in ema_sd:
                    ema_sd[name] = ema_t.to(device="cpu", dtype=ema_sd[name].dtype)
            torch.save(
                {
                    "global_step": global_step,
                    "ema_decay": args.ema_decay,
                    "model_state_dict": raw_sd,
                    "ema_model_state_dict": ema_sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                save_path,
            )
            print(f"\n>>> [Checkpoint Saved (with EMA)] -> {save_path}\n")

    print(f"\n[Completed] Fine-tuning finished! Total time: {(time.time() - start_time) / 60:.2f} min.")


if __name__ == "__main__":
    main()
