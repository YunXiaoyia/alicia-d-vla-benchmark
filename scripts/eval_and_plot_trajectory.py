#!/usr/bin/env python3
"""Evaluate all 4 models: TurboVLA vs pi0.5 vs SmolVLA vs ACT trajectory tracking on real video segment."""

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lerobot"))
sys.path.insert(0, str(REPO_ROOT / "TurboVLA"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from examples.alicia.eval_alicia_turbovla import TurboVLAAliciaPolicy, DEFAULT_WIPES_TASK


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset (Episode 0, golden grasping phase: frame 150 to 300)
    dataset_dir = REPO_ROOT / "datasets/wipes-to-green-plate-01"
    print(f"Loading dataset from {dataset_dir}...")
    ds = LeRobotDataset("wipes-to-green-plate-01", root=str(dataset_dir))
    
    start_frame = 150
    end_frame = 300
    total_frames = end_frame - start_frame
    print(f"Target evaluation window: Frame {start_frame} to {end_frame} ({total_frames} frames, {total_frames/30:.1f}s)")

    # Collect ground-truth actions
    gt_actions = np.array([ds[i]["action"].numpy() for i in range(start_frame, end_frame)])  # [T, 7]

    # Pre-build observation frames for fast iteration
    print("Pre-fetching frames...")
    cached_obs = []
    cached_batches = []
    for idx in range(start_frame, end_frame):
        item = ds[idx]
        cached_batches.append({
            "observation.state": item["observation.state"].unsqueeze(0),
            "observation.images.top": item["observation.images.top"].unsqueeze(0),
            "observation.images.wrist": item["observation.images.wrist"].unsqueeze(0),
            "task": DEFAULT_WIPES_TASK,
        })
        cached_obs.append({
            "observation.state": item["observation.state"].numpy(),
            "observation.images.top": (item["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype("uint8"),
            "observation.images.wrist": (item["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype("uint8"),
        })

    model_results = {}

    # -------------------------------------------------------------
    # Model 1: TurboVLA (Warm-start 20k)
    # -------------------------------------------------------------
    print("\n--- [1/4] Loading & Evaluating TurboVLA ---")
    turbovla_ckpt = REPO_ROOT / "weights/turbovla_wipes/turbovla_wipes_warmstart_step_20000.pth"
    stats_path = REPO_ROOT / "weights/turbovla_wipes/wipes_stats.json"
    dinov3_path = REPO_ROOT / "models/dinov3-vitb16-pretrain-lvd1689m"
    bert_path = REPO_ROOT / "models/bert-base-uncased"
    
    turbovla_policy = TurboVLAAliciaPolicy(
        ckpt_path=str(turbovla_ckpt),
        stats_path=str(stats_path),
        dinov3_path=str(dinov3_path),
        bert_path=str(bert_path),
        device=str(device),
        n_action_steps=15,
    )
    turbovla_policy.eval()
    turbovla_policy.reset()

    turbovla_actions, turbovla_lats = [], []
    for i in range(total_frames):
        if len(turbovla_policy._action_queue) == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = turbovla_policy.select_action(cached_batches[i])
            torch.cuda.synchronize()
            turbovla_lats.append((time.perf_counter() - t0) * 1000)
        else:
            act = turbovla_policy.select_action(cached_batches[i])
        turbovla_actions.append(act.squeeze(0).cpu().numpy())

    model_results["TurboVLA"] = {
        "actions": np.array(turbovla_actions),
        "lat": np.mean(turbovla_lats),
        "color": "#0984e3",  # Blue
        "style": "--",
    }
    del turbovla_policy
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Model 2: pi0.5 (A800 full 20k)
    # -------------------------------------------------------------
    print("\n--- [2/4] Loading & Evaluating pi0.5 ---")
    pi05_path = REPO_ROOT / "weights/pi05_wipes_a800_full_20k"
    pi05_cfg = PreTrainedConfig.from_pretrained(str(pi05_path))
    pi05_cfg.pretrained_path = str(pi05_path)
    pi05_rename = {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    }
    pi05_policy = make_policy(cfg=pi05_cfg, ds_meta=ds.meta, rename_map=pi05_rename)
    pi05_policy.eval()
    pi05_pre, pi05_post = make_pre_post_processors(policy_cfg=pi05_cfg, pretrained_path=str(pi05_path))
    pi05_policy.reset()
    pi05_pre.reset()
    pi05_post.reset()

    pi05_actions, pi05_lats = [], []
    for i in range(total_frames):
        queue_len = len(getattr(pi05_policy, "_action_queue", []))
        if queue_len == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = predict_action(
                cached_obs[i], pi05_policy, device, pi05_pre, pi05_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
            torch.cuda.synchronize()
            pi05_lats.append((time.perf_counter() - t0) * 1000)
        else:
            act = predict_action(
                cached_obs[i], pi05_policy, device, pi05_pre, pi05_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
        pi05_actions.append(act.squeeze(0).cpu().numpy())

    model_results["pi0.5"] = {
        "actions": np.array(pi05_actions),
        "lat": np.mean(pi05_lats),
        "color": "#d63031",  # Red
        "style": ":",
    }
    del pi05_policy, pi05_pre, pi05_post
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Model 3: SmolVLA (30k)
    # -------------------------------------------------------------
    print("\n--- [3/4] Loading & Evaluating SmolVLA ---")
    smol_path = REPO_ROOT / "weights/smolvla_wipes_ck30_bs64_30000/030000/pretrained_model"
    smol_cfg = PreTrainedConfig.from_pretrained(str(smol_path))
    smol_cfg.pretrained_path = str(smol_path)
    smol_rename = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    smol_policy = make_policy(cfg=smol_cfg, ds_meta=ds.meta, rename_map=smol_rename)
    smol_policy.eval()
    smol_pre, smol_post = make_pre_post_processors(policy_cfg=smol_cfg, pretrained_path=str(smol_path))
    smol_policy.reset()
    smol_pre.reset()
    smol_post.reset()

    smol_actions, smol_lats = [], []
    for i in range(total_frames):
        queue_len = len(getattr(smol_policy, "_action_queue", []))
        if queue_len == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = predict_action(
                cached_obs[i], smol_policy, device, smol_pre, smol_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
            torch.cuda.synchronize()
            smol_lats.append((time.perf_counter() - t0) * 1000)
        else:
            act = predict_action(
                cached_obs[i], smol_policy, device, smol_pre, smol_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
        smol_actions.append(act.squeeze(0).cpu().numpy())

    model_results["SmolVLA"] = {
        "actions": np.array(smol_actions),
        "lat": np.mean(smol_lats),
        "color": "#e67e22",  # Orange
        "style": "-.",
    }
    del smol_policy, smol_pre, smol_post
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Model 4: ACT (50k)
    # -------------------------------------------------------------
    print("\n--- [4/4] Loading & Evaluating ACT ---")
    act_path = REPO_ROOT / "weights/act_wipes_to_green_plate_run1/checkpoints/050000/pretrained_model"
    act_cfg = PreTrainedConfig.from_pretrained(str(act_path))
    act_cfg.pretrained_path = str(act_path)
    act_policy = make_policy(cfg=act_cfg, ds_meta=ds.meta)
    act_policy.eval()
    act_pre, act_post = make_pre_post_processors(policy_cfg=act_cfg, pretrained_path=str(act_path))
    act_policy.reset()
    act_pre.reset()
    act_post.reset()

    act_actions, act_lats = [], []
    for i in range(total_frames):
        queue_len = len(getattr(act_policy, "_action_queue", []))
        if queue_len == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = predict_action(
                cached_obs[i], act_policy, device, act_pre, act_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
            torch.cuda.synchronize()
            act_lats.append((time.perf_counter() - t0) * 1000)
        else:
            act = predict_action(
                cached_obs[i], act_policy, device, act_pre, act_post, False, DEFAULT_WIPES_TASK, "alicia_d_follower"
            )
        act_actions.append(act.squeeze(0).cpu().numpy())

    model_results["ACT"] = {
        "actions": np.array(act_actions),
        "lat": np.mean(act_lats),
        "color": "#16a085",  # Teal/Green
        "style": "-",
    }
    del act_policy, act_pre, act_post
    torch.cuda.empty_cache()

    # Print Summary Table
    print("\n" + "=" * 80)
    print("Trajectory Tracking Performance Summary across 4 Models (150 Frames, 5.0 Seconds):")
    print(f"{'Model':<12} | {'6-Joint MAE':<14} | {'Gripper MAE':<14} | {'Chunk Latency':<14}")
    print("-" * 80)
    for name, data in model_results.items():
        errs = np.abs(data["actions"] - gt_actions)
        j_mae = errs[:, :6].mean()
        g_mae = errs[:, 6].mean()
        data["j_mae"] = j_mae
        data["g_mae"] = g_mae
        data["errs"] = errs
        print(f"{name:<12} | {j_mae:.3f} deg        | {g_mae:.2f}          | {data['lat']:.1f} ms")
    print("=" * 80)

    # -------------------------------------------------------------
    # Plot All 4 Models
    # -------------------------------------------------------------
    joint_names = [f"Joint {i}" for i in range(1, 7)] + ["Gripper"]
    time_axis = np.arange(total_frames) / 30.0  # seconds

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=160)
    axes = axes.flatten()

    color_gt = "#2d3436"  # Ground Truth Dark Charcoal

    for i in range(6):
        ax = axes[i]
        ax.plot(time_axis, gt_actions[:, i], label="Ground Truth (Human)", color=color_gt, linewidth=2.4, linestyle="-", zorder=5)
        for m_name, m_data in model_results.items():
            err_val = m_data["errs"][:, i].mean()
            ax.plot(
                time_axis, m_data["actions"][:, i],
                label=f"{m_name} ({err_val:.2f}°)",
                color=m_data["color"],
                linewidth=1.7,
                linestyle=m_data["style"],
            )
        ax.set_title(f"{joint_names[i]} (Degrees)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Angle (°)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)

    # Gripper
    ax_grip = axes[6]
    ax_grip.plot(time_axis, gt_actions[:, 6], label="Ground Truth", color=color_gt, linewidth=2.4, zorder=5)
    for m_name, m_data in model_results.items():
        ax_grip.plot(
            time_axis, m_data["actions"][:, 6],
            label=f"{m_name} (MAE {m_data['g_mae']:.1f})",
            color=m_data["color"],
            linewidth=1.7,
            linestyle=m_data["style"],
        )
    ax_grip.set_title("Gripper Opening [0=Closed, 1000=Open]", fontsize=12, fontweight="bold")
    ax_grip.set_xlabel("Time (s)", fontsize=10)
    ax_grip.set_ylabel("Opening Units", fontsize=10)
    ax_grip.grid(True, linestyle="--", alpha=0.5)
    ax_grip.legend(fontsize=7.5, loc="best", framealpha=0.9)

    # Quantitative Comparison Bar Chart
    ax_bar = axes[7]
    models = list(model_results.keys())
    x = np.arange(len(models))
    j_maes = [model_results[m]["j_mae"] for m in models]
    colors = [model_results[m]["color"] for m in models]

    bars = ax_bar.bar(x, j_maes, color=colors, alpha=0.85, width=0.55)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax_bar.set_title("Overall 6-Joint MAE (°)", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Mean Absolute Error (°)", fontsize=10)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax_bar.annotate(f"{h:.3f}°",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Inset for Latency
    latency_summary = (
        "Inference Chunk Latency:\n" +
        "\n".join([f"• {m}: {model_results[m]['lat']:.1f} ms" for m in models])
    )
    ax_bar.text(
        0.05, 0.95, latency_summary,
        transform=ax_bar.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    )

    plt.suptitle(
        f"Benchmark of 4 Policy Architectures on Alicia-D Wipes Pick-and-Place Trajectory\n"
        f"TurboVLA vs pi0.5 vs SmolVLA vs ACT | Frame 150 -> 300 (5.0s, Horizon Chunk=15)",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = REPO_ROOT / "outputs/eval_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "trajectory_tracking_all_models.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSuccessfully generated and saved 4-model comparison curve to: {out_path}")


if __name__ == "__main__":
    main()
