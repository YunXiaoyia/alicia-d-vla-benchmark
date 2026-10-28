#!/usr/bin/env python3
"""
Generate publication-quality benchmark figures and real-robot closed-loop GIF demos
for the Alicia-D VLA Benchmark repository.

Outputs:
1. assets/benchmarks/turbovla_warmstart_vs_scratch.png
   - 20,000-step Training Loss & Gradient Norm ablation (LIBERO 40k Warm-Start vs From-Scratch)
2. assets/benchmarks/temporal_smoothing_ablation.png
   - Real-robot closed-loop trajectory smoothness comparison (No Temporal Ensemble vs ACTTemporalEnsembler vs RTC)
3. assets/hardware/camera_views_overview.png
   - Dual-camera observation overview (Top Global View + Wrist Eye-in-Hand D405 View)
4. assets/demos/rollout_pi05_rtc.gif
   - Real-robot closed-loop execution of Pi0.5 (A800 20k + RTC n_action_steps=15)
5. assets/demos/rollout_smolvla_ensemble.gif
   - Real-robot closed-loop execution of SmolVLA (30k + ACTTemporalEnsembler)
6. assets/demos/rollout_act50_ensemble.gif
   - Real-robot closed-loop execution of ACT (50k + Temporal Ensemble)
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = "/media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark"
EVAL_RUNS_ROOT = "/media/yunyi/Data/Desktop/work/vla/eval_runs"


def parse_training_log(log_path):
    steps, losses, lrs, grads = [], [], [], []
    pattern = re.compile(
        r"Step\s+(\d+)/\d+\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*HeadLR:\s*([0-9.eE+-]+)\s*\|\s*GradNorm:\s*([0-9.]+)"
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                grads.append(float(m.group(4)))
    return np.array(steps), np.array(losses), np.array(lrs), np.array(grads)


def plot_turbovla_warmstart_ablation():
    print("[1/4] Generating TurboVLA Warm-Start vs From-Scratch ablation plot...")
    warm_log = os.path.join(REPO_ROOT, "docs/train_wipes_a_warmstart.log")
    scratch_log = os.path.join(REPO_ROOT, "docs/train_wipes_b_scratch.log")

    s_w, l_w, _, g_w = parse_training_log(warm_log)
    s_s, l_s, _, g_s = parse_training_log(scratch_log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")

    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.35)

    # Panel 1: Training Loss (Step 1000 -> 20000 for clear scale)
    mask_w = s_w >= 800
    mask_s = s_s >= 800
    ax1.plot(s_s[mask_s], l_s[mask_s], color="#EF4444", linewidth=2.2, alpha=0.9,
             label=f"From Scratch (Cold Start) — Final Loss: {l_s[-1]:.4f}")
    ax1.plot(s_w[mask_w], l_w[mask_w], color="#10B981", linewidth=2.4, alpha=0.95,
             label=f"LIBERO 40k Warm-Start (12→30 Query Interp.) — Final Loss: {l_w[-1]:.4f} (-10.2%)")

    ax1.set_title("Action Chunk L1 Loss Convergence (20,000 Steps, Global BS=128)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Smooth L1 Loss", fontsize=11)
    ax1.set_ylim(0.004, 0.045)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    # Zoom-in inset for final 10k -> 20k steps
    axins = inset_axes(ax1, width="46%", height="44%", loc="center right", borderpad=1.8)
    z_w = s_w >= 10000
    z_s = s_s >= 10000
    axins.plot(s_s[z_s], l_s[z_s], color="#EF4444", linewidth=2.0)
    axins.plot(s_w[z_w], l_w[z_w], color="#10B981", linewidth=2.2)
    axins.fill_between(s_w[z_w], l_w[z_w], l_s[z_s], color="#10B981", alpha=0.15)
    axins.set_title("Zoom: Steps 10k → 20k (0.0059 vs 0.0053)", fontsize=8.5, fontweight="bold")
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.tick_params(labelsize=8)

    # Panel 2: Gradient Norm Stability
    ax2.plot(s_s, g_s, color="#EF4444", linewidth=1.6, alpha=0.75,
             label=f"From Scratch GradNorm (Max: {g_s.max():.2f}, Mean: {g_s.mean():.2f})")
    ax2.plot(s_w, g_w, color="#10B981", linewidth=1.8, alpha=0.9,
             label=f"LIBERO 40k Warm-Start GradNorm (Max: {g_w.max():.2f}, Mean: {g_w.mean():.2f})")
    ax2.set_title("Gradient Norm Stability Across Fine-Tuning", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Gradient L2 Norm", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    fig.suptitle(
        "TurboVLA Cross-Embodiment Transfer Ablation on Alicia-D (71 Episodes / 24,535 Frames)",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(REPO_ROOT, "assets/benchmarks/turbovla_warmstart_vs_scratch.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> Saved to {out_path}")


def plot_temporal_smoothing_ablation():
    print("[2/4] Generating Real-Robot Temporal Smoothing & Chunk Boundary Ablation plot...")
    runs = [
        ("SmolVLA (n_action_steps=30, No Temporal Smoothing)", "yunyi/n30_no_temporal", "#EF4444"),
        ("SmolVLA + ACTTemporalEnsembler (coeff=0.01)", "yunyi/eval-wipes-ensemble-acm6", "#10B981"),
        ("π0.5 (A800 20k + RTC Prefix Guidance n=15)", "eval-wipes-pi05-a800-20k-rtc-ns15", "#3B82F6"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")

    for col, (title, rel_dir, color) in enumerate(runs):
        pq_path = os.path.join(EVAL_RUNS_ROOT, rel_dir, "data/chunk-000/file-000.parquet")
        df = pd.read_parquet(pq_path)
        actions = np.stack(df["action"].values)        # [N, 7]
        states = np.stack(df["observation.state"].values)  # [N, 7]
        # Focus on the first 300 frames (10 seconds of active grasping)
        T = min(300, len(actions))
        t_sec = np.arange(T) / 30.0

        # Joint 2 (Shoulder Lift, idx=1) & Joint 3 (Elbow, idx=2)
        act_j2 = actions[:T, 1]
        st_j2 = states[:T, 1]
        # Step-to-step action velocity change (acceleration / boundary jump proxy) across 6 arm joints
        arm_diff = np.linalg.norm(np.diff(actions[:T, :6], axis=0), axis=1)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.set_facecolor("#FFFFFF")
        ax_bot.set_facecolor("#FFFFFF")

        ax_top.plot(t_sec, st_j2, color="#1F2937", linewidth=1.8, linestyle="--", alpha=0.8, label="Measured State (Joint 2)")
        ax_top.plot(t_sec, act_j2, color=color, linewidth=2.0, alpha=0.95, label="Commanded Action (Joint 2)")
        ax_top.set_title(title, fontsize=11, fontweight="bold")
        ax_top.set_xlabel("Real-Robot Execution Time (s)", fontsize=9.5)
        ax_top.set_ylabel("Joint 2 Angle (°)", fontsize=10)
        ax_top.grid(True, linestyle="--", alpha=0.35)
        ax_top.legend(loc="best", fontsize=8.5)

        # Highlight 30-step chunk boundaries on top & bottom
        step_interval = 15 if "rtc" in rel_dir else 30
        for b in range(step_interval, T, step_interval):
            ax_top.axvline(b / 30.0, color="#9CA3AF", linestyle=":", linewidth=0.8, alpha=0.6)
            ax_bot.axvline(b / 30.0, color="#9CA3AF", linestyle=":", linewidth=0.8, alpha=0.6)

        p95 = np.percentile(arm_diff, 95)
        max_jump = arm_diff.max()
        ax_bot.plot(t_sec[1:], arm_diff, color=color, linewidth=1.5, alpha=0.9,
                    label=f"6-DoF Step Jump ‖Δa_t‖₂ (Max: {max_jump:.2f}°, P95: {p95:.2f}°)")
        ax_bot.set_title(f"Step-to-Step Action Discontinuity (Max={max_jump:.2f}°)", fontsize=10.5, fontweight="bold")
        ax_bot.set_xlabel("Real-Robot Execution Time (s)", fontsize=9.5)
        ax_bot.set_ylabel("‖a_t - a_{t-1}‖₂ (°/step)", fontsize=10)
        ax_bot.set_ylim(0, max(12.0, max_jump * 1.12))
        ax_bot.grid(True, linestyle="--", alpha=0.35)
        ax_bot.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Real-Robot Closed-Loop Smoothness Ablation: Suppressing Action Chunk Boundary Jumps on Alicia-D",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(REPO_ROOT, "assets/benchmarks/temporal_smoothing_ablation.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> Saved to {out_path}")


def plot_camera_views_overview():
    print("[3/4] Generating Dual-Camera Hardware Observation Overview...")
    top_img_path = os.path.join(REPO_ROOT, "assets/hardware/dataset_top.png")
    wrist_img_path = os.path.join(REPO_ROOT, "assets/hardware/dataset_wrist.png")

    top_img = Image.open(top_img_path).convert("RGB")
    wrist_img = Image.open(wrist_img_path).convert("RGB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")

    ax1.imshow(top_img)
    ax1.set_title("Global Third-Person Camera\n(observation.images.top, 640×480 @ 30Hz)", fontsize=10.5, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2.imshow(wrist_img)
    ax2.set_title("Eye-in-Hand Wrist Camera (RealSense D405)\n(observation.images.wrist, 640×480 @ 30Hz)", fontsize=10.5, fontweight="bold", pad=8)
    ax2.axis("off")

    fig.suptitle(
        "Alicia-D 6-DoF Real-Robot Dual-Camera Observation Setup",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = os.path.join(REPO_ROOT, "assets/hardware/camera_views_overview.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> Saved to {out_path}")


def make_rollout_gif(run_rel_dir, title_text, out_gif_name, stride=6, max_frames=450):
    pq_path = os.path.join(EVAL_RUNS_ROOT, run_rel_dir, "data/chunk-000/file-000.parquet")
    top_dir = os.path.join(EVAL_RUNS_ROOT, run_rel_dir, "images/observation.images.top/episode-000000")
    wrist_dir = os.path.join(EVAL_RUNS_ROOT, run_rel_dir, "images/observation.images.wrist/episode-000000")

    df = pd.read_parquet(pq_path)
    actions = np.stack(df["action"].values)
    states = np.stack(df["observation.state"].values)

    top_files = sorted(glob.glob(os.path.join(top_dir, "*.png")))
    wrist_files = sorted(glob.glob(os.path.join(wrist_dir, "*.png")))
    N = min(len(top_files), len(wrist_files), len(actions), max_frames)

    indices = list(range(0, N, stride))
    frames_pil = []

    # Pre-compute time axis
    t_all = np.arange(N) / 30.0

    for idx in indices:
        top_im = Image.open(top_files[idx]).convert("RGB").resize((320, 240))
        wrist_im = Image.open(wrist_files[idx]).convert("RGB").resize((320, 240))

        # Render right-side trajectory plot with matplotlib
        fig, (ax_j, ax_g) = plt.subplots(2, 1, figsize=(3.6, 2.4), dpi=100, gridspec_kw={"height_ratios": [1.5, 1.0]})
        fig.patch.set_facecolor("#111827")
        for ax in (ax_j, ax_g):
            ax.set_facecolor("#1F2937")
            ax.tick_params(colors="#D1D5DB", labelsize=6.5)
            for spine in ax.spines.values():
                spine.set_color("#374151")

        # Plot joints 1, 2, 3 up to current frame
        colors = ["#60A5FA", "#34D399", "#FBBF24"]
        labels = ["J1", "J2", "J3"]
        for j_i, (c, lb) in enumerate(zip(colors, labels)):
            ax_j.plot(t_all[:N], states[:N, j_i], color=c, alpha=0.22, linewidth=1.0)
            ax_j.plot(t_all[:idx+1], actions[:idx+1, j_i], color=c, alpha=0.95, linewidth=1.4, label=lb)
        ax_j.axvline(t_all[idx], color="#F87171", linestyle="--", linewidth=1.0)
        ax_j.set_title(f"6-DoF Joint Cmd (t={t_all[idx]:.1f}s)", color="#F9FAFB", fontsize=7.5, fontweight="bold", pad=2)
        ax_j.legend(loc="upper right", fontsize=5.5, ncol=3, facecolor="#1F2937", edgecolor="none", labelcolor="#F9FAFB")
        ax_j.set_xlim(0, t_all[N - 1])

        # Plot gripper
        ax_g.plot(t_all[:N], states[:N, 6], color="#A78BFA", alpha=0.25, linewidth=1.0)
        ax_g.plot(t_all[:idx+1], actions[:idx+1, 6], color="#A78BFA", alpha=0.95, linewidth=1.5)
        ax_g.axvline(t_all[idx], color="#F87171", linestyle="--", linewidth=1.0)
        ax_g.set_title("Gripper (0=Close, 1000=Open)", color="#F9FAFB", fontsize=7.0, pad=2)
        ax_g.set_xlim(0, t_all[N - 1])
        ax_g.set_ylim(-50, 1080)

        plt.tight_layout(pad=0.35)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plot_im = Image.fromarray(buf).resize((360, 240))
        plt.close(fig)

        # Stitch: Header (28px) + [Top (320x240) | Wrist (320x240) | Plot (360x240)] = 1000 x 268
        canvas = Image.new("RGB", (1000, 268), (17, 24, 39))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{title_text} | {t_all[idx]:.1f}s ({idx:03d}/{N:03d})", fill=(249, 250, 251))
        draw.text((330, 8), "Eye-in-Hand Wrist Camera (D405)", fill=(156, 163, 175))
        draw.text((655, 8), "Real-Time Closed-Loop Action & State", fill=(156, 163, 175))

        canvas.paste(top_im, (0, 28))
        canvas.paste(wrist_im, (320, 28))
        canvas.paste(plot_im, (640, 28))
        frames_pil.append(canvas)

    out_path = os.path.join(REPO_ROOT, "assets/demos", out_gif_name)
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=80,  # ~2.5x playback speed for crisp README viewing
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  -> Saved GIF {out_path} ({len(frames_pil)} frames, {size_mb:.2f} MB)")


def main():
    plot_turbovla_warmstart_ablation()
    plot_temporal_smoothing_ablation()
    plot_camera_views_overview()

    print("[4/4] Generating Real-Robot Closed-Loop Triple-View GIFs...")
    make_rollout_gif(
        "eval-wipes-pi05-a800-20k-rtc-ns15",
        "Pi0.5 (20k + RTC n=15)",
        "rollout_pi05_rtc.gif",
        stride=5,
        max_frames=347,
    )
    make_rollout_gif(
        "yunyi/eval-wipes-ensemble-acm6",
        "SmolVLA (30k + Ensembler)",
        "rollout_smolvla_ensemble.gif",
        stride=8,
        max_frames=560,
    )
    make_rollout_gif(
        "yunyi/eval-wipes-act50-ensemble-ep1",
        "ACT (50k + Ensembler)",
        "rollout_act50_ensemble.gif",
        stride=8,
        max_frames=560,
    )
    print("All visual assets generated successfully!")


if __name__ == "__main__":
    main()
