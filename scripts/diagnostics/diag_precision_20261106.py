"""Offline precision and EMA audit; never connects to robot hardware."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from turbovla.models.turbovla import build_turbovla
from experiments.wipes.wipes_dataset import LeRobotWipesDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yunyi/Desktop/vla")
    args = ap.parse_args()
    root = Path(args.root)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    checkpoints = [
        root / "TurboVLA/outputs/turbovla_wipes_freeze_vit/turbovla_wipes_freeze_vit_step_08000.pth",
        root / "TurboVLA/outputs/turbovla_wipes_warm_full/turbovla_wipes_warm_full_step_12500.pth",
    ]
    for path in checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        saved = ck.get("args", {})
        cfg = dict(saved)
        cfg.update(
            dinov3_path=str(root / "models/dinov3-vitb16-pretrain-lvd1689m"),
            bert_path=str(root / "models/bert-base-uncased"),
            max_text_len=256,
            text_padding_length=saved.get("text_padding_length", 21),
            text_padding_length_by_instruction={},
            zero_padded_text=saved.get("padding_strategy", "key_padding_mask") == "zero_fill",
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.1,
            freeze_vision_encoder=True,
            freeze_text_encoder=True,
            dinov3_precision="bf16_autocast",
            local_files_only=True,
            num_views=2,
            image_size=saved.get("expected_image_size", 256),
        )
        if cfg.get("text_padding_length", 21) <= 0:
            cfg["text_padding_length"] = None
        model = build_turbovla(SimpleNamespace(**cfg)).cuda().eval()
        ds = LeRobotWipesDataset(
            dataset_dir=str(root / "datasets/wipes-to-green-plate-01"),
            stats_path=str(root / "TurboVLA/experiments/wipes/wipes_stats.json"),
            state_dim=saved.get("state_dim", 8),
            state_norm_mode=saved.get("state_norm_mode", "zscore"),
            chunk_size=saved.get("chunk_size", 30),
            expected_image_size=saved.get("expected_image_size", 256),
        )
        ids = []
        for ep in range(0, 71, 10):
            row = ds.lerobot_ds.meta.episodes[ep]
            lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            st = ds.all_states[lo:hi]
            active = np.flatnonzero(np.max(np.abs(st[:, :6] - st[0, :6]), axis=1) > 1.5)
            first = int(active[0]) if len(active) else 0
            for offset in (0, 40, 80, 130):
                ids.append((ep, offset, min(lo + first + offset, hi - 1)))
        items = [ds[idx] for _, _, idx in ids]
        action_min = torch.tensor(ds.action_min, device="cuda")
        action_range = torch.tensor(ds.action_max - ds.action_min, device="cuda")

        def predict(outer_bf16=False, permute=False):
            output = []
            hook = None
            if permute:
                def shuffle(module, inputs, tokens):
                    return tokens[:, :, torch.randperm(tokens.shape[2], device=tokens.device)]
                hook = model.vision_encoder.register_forward_hook(shuffle)
            try:
                with torch.inference_mode():
                    for images, text, state, gt, mask in items:
                        samples = {"dinov3": torch.stack([images[0]["dinov3"], images[1]["dinov3"]])[None].cuda()}
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=outer_bf16):
                            pred = model([text], samples, state[None].cuda())
                        raw = (pred.float().clamp(-1, 1) + 1) * 0.5 * (action_range + 1e-8) + action_min
                        output.append(raw[0].cpu().numpy())
            finally:
                if hook is not None:
                    hook.remove()
            return np.stack(output)

        def metrics(pred):
            gt = np.stack([((it[3].numpy() + 1) * 0.5 * (ds.action_max - ds.action_min + 1e-8) + ds.action_min) for it in items])
            mask = np.stack([it[4].numpy() for it in items]).astype(bool)
            error = np.abs(pred - gt)
            return {
                "c0_mae": error[:, 0].mean(0).tolist(),
                "c29_mae": error[:, -1][mask[:, -1]].mean(0).tolist(),
                "valid_chunk_mae": error[mask].mean(0).tolist(),
                "by_offset_c0_arm_mae": {str(offset): float(error[[i for i, row in enumerate(ids) if row[1] == offset], 0, :6].mean()) for offset in (0, 40, 80, 130)},
            }

        report = {"checkpoint": str(path), "args": saved, "global_step": ck.get("global_step"), "frames": ids}
        load = model.load_state_dict(ck["ema_model_state_dict"], strict=True)
        ema_fp32 = predict()
        ema_bf16 = predict(True)
        difference = np.abs(ema_fp32 - ema_bf16)
        report["ema_fp32"] = metrics(ema_fp32)
        report["ema_bf16"] = metrics(ema_bf16)
        report["precision_difference"] = {"mae": difference.mean((0, 1)).tolist(), "max": difference.max((0, 1)).tolist()}
        permuted = predict(permute=True)
        report["patch_permutation_max_difference"] = np.abs(permuted - ema_fp32).max((0, 1)).tolist()
        if "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"], strict=True)
            raw_fp32 = predict()
            report["raw_fp32"] = metrics(raw_fp32)
        print("RESULT " + json.dumps(report), flush=True)
        del model, ds, ck, items
        gc.collect()
        torch.cuda.empty_cache()
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
