"""Compare TurboVLA predictions to a recorded ACT rollout without robot access."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as TF

from turbovla.models.turbovla import build_turbovla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yunyi/Desktop/vla")
    ap.add_argument("--observations", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.observations)
    stats = json.loads((root / "TurboVLA/experiments/wipes/wipes_stats.json").read_text())["wipes"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    action_min = torch.tensor(stats["action"]["min"], device="cuda")
    action_max = torch.tensor(stats["action"]["max"], device="cuda")
    report = {"reference": str(data["reference_kind"]), "warning": "ACT executed future commands are a policy reference, not expert ground truth.", "models": {}}
    for tag, suffix in (("freeze_vit", "08000"), ("warm_full", "12500")):
        path = root / f"TurboVLA/outputs/turbovla_wipes_{tag}/turbovla_wipes_{tag}_step_{suffix}.pth"
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
        model = build_turbovla(SimpleNamespace(**cfg)).cuda().eval()
        model.load_state_dict(ck["ema_model_state_dict"], strict=True)
        state_dim = saved.get("state_dim", 8)
        pmean = torch.tensor(stats["proprio"]["mean"][:state_dim], device="cuda")
        pstd = torch.tensor(stats["proprio"]["std"][:state_dim], device="cuda")
        pmin = torch.tensor(stats["proprio"]["min"][:state_dim], device="cuda")
        pmax = torch.tensor(stats["proprio"]["max"][:state_dim], device="cuda")
        predictions = []
        with torch.inference_mode():
            for i in range(len(data["frame_index"])):
                images = []
                for view in ("top", "wrist"):
                    image = torch.from_numpy(data[view][i].copy()).permute(2, 0, 1)[None].cuda().float() / 255.0
                    image = TF.resize(image, [cfg["image_size"], cfg["image_size"]], antialias=True)
                    images.append((image - mean) / std)
                samples = {"dinov3": torch.stack(images, dim=1)}
                state = torch.from_numpy(data["state"][i].copy())[None].cuda()
                if state_dim == 8:
                    state = torch.nn.functional.pad(state, (0, 1))
                else:
                    state = state[:, :state_dim]
                if saved.get("state_norm_mode", "zscore") == "min_max":
                    state = (2 * (state - pmin) / (pmax - pmin + 1e-8) - 1).clamp(-1, 1)
                    if state_dim == 8:
                        state[:, 7] = 0
                else:
                    state = (state - pmean) / (pstd + 1e-8)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    norm = model(["Pick up the pack of cleaning wipes and place it on the green plate"], samples, state)
                raw = (norm.float().clamp(-1, 1) + 1) * 0.5 * (action_max - action_min + 1e-8) + action_min
                predictions.append(raw[0].cpu().numpy())
        pred = np.stack(predictions)
        error = np.abs(pred - data["future_actions"])
        mask = data["future_action_mask"]
        records = []
        for i, frame in enumerate(data["frame_index"]):
            records.append({
                "frame": int(frame),
                "state": data["state"][i].tolist(),
                "prediction_c0": pred[i, 0].tolist(),
                "prediction_c29": pred[i, -1].tolist(),
                "c0_minus_state": (pred[i, 0] - data["state"][i]).tolist(),
                "c29_minus_state": (pred[i, -1] - data["state"][i]).tolist(),
                "act_reference_c0": data["future_actions"][i, 0].tolist(),
                "act_reference_c29": data["future_actions"][i, -1].tolist(),
                "arm_reference_chunk_mae": float(error[i, mask[i], :6].mean()),
            })
        metrics = {
            "reference_chunk_mae_per_joint": error[mask].mean(0).tolist(),
            "c0_state_jump_mean_per_joint": np.abs(pred[:, 0] - data["state"]).mean(0).tolist(),
            "c0_state_jump_max_per_joint": np.abs(pred[:, 0] - data["state"]).max(0).tolist(),
            "frames": records,
        }
        report["models"][tag] = metrics
        np.savez_compressed(output / f"{tag}_predictions.npz", prediction=pred, frame_index=data["frame_index"], state=data["state"], act_reference=data["future_actions"], act_reference_mask=mask)
        print("RESULT " + json.dumps({"model": tag, **metrics}), flush=True)
        del model, ck
        gc.collect()
        torch.cuda.empty_cache()
    (output / "real_rollout_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
