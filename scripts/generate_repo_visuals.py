#!/usr/bin/env python3
"""Generate publication-grade benchmark figures and real-robot rollout GIFs for alicia-d-vla-benchmark.

Generates:
1. `assets/hardware/camera_views_overview.png` — Synchronized Dual-Camera (`top` + `wrist`) across 4 task stages
2. `assets/benchmarks/turbovla_warmstart_vs_scratch.png` — 3-Way Cross-Embodiment Transfer & Frozen-DINOv3 Ablation
3. `assets/benchmarks/temporal_smoothing_ablation.png` — 4-Model Real-Robot Closed-Loop Temporal Smoothing Ablation
   (`ACT`, `SmolVLA`, `π0.5`, `TurboVLA` — `No Smooth` vs `Smooth` on physical Alicia-D hardware)
4. `assets/demos/rollout_*.gif` — 4 Real-Robot Closed-Loop Rollout GIFs (`TurboVLA`, `π0.5`, `SmolVLA`, `ACT`)
"""

import glob
import os
import re
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path("/media/yunyi/Data/Desktop/work/vla")
BENCH_ROOT = WORKSPACE / "alicia-d-vla-benchmark"
ASSETS_DIR = BENCH_ROOT / "assets"
LOCAL_EVAL_DIR = WORKSPACE / "eval_runs/local"


def parse_turbovla_log(log_path: Path):
    steps, losses, grad_norms, lrs, speeds = [], [], [], [], []
    pattern = re.compile(
        r"Step\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+HeadLR:\s+([\d.eE+-]+)\s+\|\s+GradNorm:\s+([\d.]+)\s+\|\s+Speed:\s+([\d.]+)"
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                grad_norms.append(float(m.group(4)))
                speeds.append(float(m.group(5)))
    return (
        np.array(steps),
        np.array(losses),
        np.array(grad_norms),
        np.array(lrs),
        np.array(speeds),
    )


def plot_turbovla_warmstart_vs_scratch():
    print("[1/4] Generating TurboVLA 3-Way Cross-Embodiment & Frozen-DINOv3 Ablation plot...")
    log_a = BENCH_ROOT / "docs/train_wipes_a_warmstart.log"
    log_b = BENCH_ROOT / "docs/train_wipes_b_scratch.log"
    log_c = BENCH_ROOT / "docs/train_wipes_freeze_vit.log"

    s_a, l_a, g_a, _, sp_a = parse_turbovla_log(log_a)
    s_b, l_b, g_b, _, sp_b = parse_turbovla_log(log_b)
    s_c, l_c, g_c, _, sp_c = parse_turbovla_log(log_c)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), dpi=180)
    plt.subplots_adjust(wspace=0.25)

    # Panel 1: L1 Action Loss Convergence (Log Scale)
    ax1 = axes[0]
    ax1.plot(s_b, l_b, color="#e17055", alpha=0.22, linewidth=1.0)
    ax1.plot(s_a, l_a, color="#0984e3", alpha=0.22, linewidth=1.0)
    ax1.plot(s_c, l_c, color="#00b894", alpha=0.25, linewidth=1.0)

    def smooth(y, box_pts=7):
        if len(y) < box_pts:
            return y
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode="valid")

    w = 7
    ax1.plot(
        s_b[w - 1 :],
        smooth(l_b, w),
        color="#d63031",
        linewidth=2.2,
        linestyle="--",
        label=f"Run B: From Scratch (20k: {l_b[-1]:.4f}, 0% Real)",
    )
    ax1.plot(
        s_a[w - 1 :],
        smooth(l_a, w),
        color="#0984e3",
        linewidth=2.2,
        linestyle="-.",
        label=f"Run A: Unfrozen LIBERO Warm-Start (20k: {l_a[-1]:.4f}, 0% Real)",
    )
    ax1.plot(
        s_c[w - 1 :],
        smooth(l_c, w),
        color="#00b894",
        linewidth=2.8,
        linestyle="-",
        label=f"Run C: Frozen-DINOv3 + Warm-Start (8k: {l_c[-1]:.4f}, 100% Real)",
    )
    ax1.set_yscale("log")
    ax1.set_title("TurboVLA Action L1 Loss Convergence (Log Scale)", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Training Steps (Effective Batch Size = 64)", fontsize=10.5)
    ax1.set_ylabel("L1 Action Loss (Log Scale)", fontsize=10.5)
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.95)

    # Panel 2: Gradient Norm Stability
    ax2 = axes[1]
    ax2.plot(
        s_b[w - 1 :],
        smooth(g_b, w),
        color="#d63031",
        linewidth=2.0,
        linestyle="--",
        label=f"Run B: Scratch (Mean GradNorm: {g_b.mean():.2f})",
    )
    ax2.plot(
        s_a[w - 1 :],
        smooth(g_a, w),
        color="#0984e3",
        linewidth=2.0,
        linestyle="-.",
        label=f"Run A: Unfrozen Warm-Start (Mean: {g_a.mean():.2f})",
    )
    ax2.plot(
        s_c[w - 1 :],
        smooth(g_c, w),
        color="#00b894",
        linewidth=2.6,
        linestyle="-",
        label=f"Run C: Frozen-DINOv3 Warm-Start (Mean: {g_c.mean():.2f})",
    )
    ax2.set_title("Gradient Norm Trajectory (7-Step Moving Avg)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=10.5)
    ax2.set_ylabel("L2 Gradient Norm", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    # Panel 3: Training Speed & Wall-Clock Time Comparison + Real-Robot Transfer Outcome
    ax3 = axes[2]
    labels = [
        "Run B: Scratch\n(Unfrozen ViT\n20k Steps)",
        "Run A: Warm-Start\n(Unfrozen ViT\n20k Steps)",
        "Run C: Warm-Start\n(Frozen Real ViT\n8k Steps)",
    ]
    wall_mins = [162.8, 162.4, 23.0]
    speeds_val = [sp_b.mean(), sp_a.mean(), sp_c.mean()]
    bar_colors = ["#d63031", "#0984e3", "#00b894"]

    x = np.arange(len(labels))
    bars = ax3.bar(x, wall_mins, width=0.45, color=bar_colors, edgecolor="#2d3436", linewidth=1.2, alpha=0.88)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.8, fontweight="bold")
    ax3.set_ylabel("Total Training Wall-Clock Time (Minutes on RTX 4090)", fontsize=10.5)
    ax3.set_title("Training Efficiency & Real-Robot Transfer Outcome", fontsize=12.5, fontweight="bold")
    ax3.set_ylim(0, 215)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)

    annotations = [
        f"{wall_mins[0]:.0f} min ({speeds_val[0]:.2f} it/s)\n0% Real (Visual Overfit)",
        f"{wall_mins[1]:.0f} min ({speeds_val[1]:.2f} it/s)\n0% Real (Sim-DINO Drift)",
        f"{wall_mins[2]:.0f} min ({speeds_val[2]:.2f} it/s, 7.1x Faster)\n100% Real-Robot Success!",
    ]
    for bar, ann in zip(bars, annotations):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4.5,
            ann,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#2d3436",
        )

    plt.suptitle(
        "TurboVLA Cross-Embodiment Transfer & Frozen-Backbone Ablation (LIBERO Simulation -> Alicia-D Real Hardware)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    out_path = ASSETS_DIR / "benchmarks/turbovla_warmstart_vs_scratch.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"   Saved -> {out_path}")


def load_real_eval_run(run_name: str):
    pq_files = sorted(glob.glob(str(LOCAL_EVAL_DIR / run_name / "data/**/*.parquet"), recursive=True))
    if not pq_files:
        raise FileNotFoundError(f"No parquet found for {run_name}")
    table = pq.read_table(pq_files[0])
    act_deg = np.stack(table["action"].to_pylist())[:, :6]
    st_deg = np.stack(table["observation.state"].to_pylist())[:, :6]
    act_rad = act_deg * (np.pi / 180.0)

    vel_deg = np.diff(act_deg, axis=0)
    acc_deg = np.diff(vel_deg, axis=0)
    jerk_deg = np.diff(acc_deg, axis=0)
    l2_rad = np.linalg.norm(np.diff(act_rad, axis=0), axis=1)

    v0, v1 = vel_deg[:-1], vel_deg[1:]
    active = (np.abs(v0) > 0.02) & (np.abs(v1) > 0.02)
    flips = ((v0 * v1) < 0) & active
    sfr = float(flips.sum()) / max(int(active.sum()), 1) * 100.0

    return {
        "act_deg": act_deg,
        "st_deg": st_deg,
        "vel_deg": vel_deg,
        "l2_rad": l2_rad,
        "l2_rad_mean": float(np.mean(l2_rad)),
        "l2_rad_p95": float(np.percentile(l2_rad, 95)),
        "sfr": sfr,
        "mean_acc": float(np.mean(np.abs(acc_deg))),
        "mean_jerk": float(np.mean(np.abs(jerk_deg))),
        "max_jump": float(np.max(np.abs(vel_deg))),
    }


def plot_temporal_smoothing_ablation():
    print("[2/4] Generating 4-Model Real-Robot Temporal Smoothing Ablation plot...")
    models_cfg = [
        ("ACT (50k)", "eval_act_no_smooth", "eval_act_smooth", "Temporal Ensemble (λ=0.01)"),
        ("SmolVLA (30k)", "eval_smolvla_no_smooth", "eval_smolvla_smooth", "Flow-Matching RTC"),
        ("π0.5 (20k)", "eval_pi05_no_smooth", "eval_pi05_smooth", "Flow-Matching RTC"),
        ("TurboVLA (Frozen-ViT 8k)", "eval_turbovla_no_smooth", "eval_turbovla_smooth", "Receding-Horizon Quintic C²"),
    ]

    data = {}
    for label, r_no, r_sm, method in models_cfg:
        data[label] = {
            "no": load_real_eval_run(r_no),
            "sm": load_real_eval_run(r_sm),
            "method": method,
        }

    fig = plt.figure(figsize=(20, 9.8), dpi=180)
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.26)

    # Top Row: Joint-Space Step-to-Step Velocity / Trajectory Smoothness for Each of the 4 Models
    for idx, (label, _, _, method) in enumerate(models_cfg):
        ax = fig.add_subplot(gs[0, idx])
        d_no = data[label]["no"]
        d_sm = data[label]["sm"]

        w = min(240, len(d_no["l2_rad"]), len(d_sm["l2_rad"]))
        t_axis = np.arange(w) / 30.0
        ax.plot(
            t_axis,
            d_no["l2_rad"][:w],
            color="#d63031",
            alpha=0.78,
            linewidth=1.35,
            label=f"Unsmoothed (Mean: {d_no['l2_rad_mean']:.4f} rad)",
        )
        ax.plot(
            t_axis,
            d_sm["l2_rad"][:w],
            color="#00b894",
            alpha=0.95,
            linewidth=1.85,
            label=f"Smoothed (Mean: {d_sm['l2_rad_mean']:.4f} rad)",
        )
        ax.set_title(f"{label}\n[{method}]", fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9.5)
        ax.set_ylabel("Step Action Delta ||Δa_t||₂ (rad)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    # Bottom Row: 4 Quantitative Comparison Bar Charts across All 4 Models
    model_names = [m[0].split(" (")[0] for m in models_cfg]
    x = np.arange(len(model_names))
    width = 0.35

    metrics_panels = [
        ("l2_rad_mean", "Mean Step Delta ||Δa_t||₂ (rad)", "Step-to-Step Action Variation (Lower = Smoother)", "{:.4f}"),
        ("mean_jerk", "Mean Joint Jerk (°/step³)", "3rd-Order Joint Jerk (Lower = Less Vibration)", "{:.3f}"),
        ("sfr", "Velocity Sign-Flip Rate (%)", "High-Freq Velocity Reversal Rate (%)", "{:.1f}%"),
        ("max_jump", "Max Single-Step Jump (°)", "Worst Chunk-Boundary Step Jump (°)", "{:.2f}°"),
    ]

    for p_idx, (m_key, ylabel, title, fmt) in enumerate(metrics_panels):
        ax = fig.add_subplot(gs[1, p_idx])
        vals_no = [data[m[0]]["no"][m_key] for m in models_cfg]
        vals_sm = [data[m[0]]["sm"][m_key] for m in models_cfg]

        b1 = ax.bar(
            x - width / 2,
            vals_no,
            width,
            label="Unsmoothed (Open-Loop Chunks)",
            color="#e17055",
            edgecolor="#2d3436",
            linewidth=1.0,
        )
        b2 = ax.bar(
            x + width / 2,
            vals_sm,
            width,
            label="Smoothed (Ensembling / RTC / Quintic C²)",
            color="#00b894",
            edgecolor="#2d3436",
            linewidth=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        ax.set_ylim(0, max(vals_no + vals_sm) * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        if p_idx == 0:
            ax.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(vals_no + vals_sm) * 0.02,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
            )

    plt.suptitle(
        "Real-Robot Closed-Loop Temporal Smoothing Ablation Across All 4 Architectures (30 Hz on Alicia-D)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    out_path = ASSETS_DIR / "benchmarks/temporal_smoothing_ablation.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"   Saved -> {out_path}")


def decode_av1_mp4(video_path: Path) -> list[np.ndarray]:
    container = av.open(str(video_path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def generate_camera_overview():
    print("[3/4] Generating Dual-Camera Overview figure from real-robot TurboVLA rollout...")
    run_dir = LOCAL_EVAL_DIR / "eval_turbovla_smooth"
    top_frames = decode_av1_mp4(run_dir / "videos/observation.images.top/chunk-000/file-000.mp4")
    wrist_frames = decode_av1_mp4(run_dir / "videos/observation.images.wrist/chunk-000/file-000.mp4")

    # Select 4 key stages from the 711-frame successful rollout
    stages = [
        (35, "Stage 1: Pre-Grasp Approach"),
        (115, "Stage 2: Precision Wipes Grasp"),
        (185, "Stage 3: Lift & Cross-Table Transport"),
        (255, "Stage 4: Green Plate Placement"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), dpi=160)
    plt.subplots_adjust(wspace=0.06, hspace=0.14)

    for col, (idx, label) in enumerate(stages):
        idx = min(idx, len(top_frames) - 1, len(wrist_frames) - 1)
        axes[0, col].imshow(top_frames[idx])
        axes[0, col].set_title(f"{label}\n[Top Global Camera]", fontsize=10.5, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(wrist_frames[idx])
        axes[1, col].set_title(f"[Wrist Eye-in-Hand Camera] (t={idx/30.0:.1f}s)", fontsize=10, color="#2d3436")
        axes[1, col].axis("off")

    plt.suptitle(
        "Alicia-D 6-DoF Dual-Camera Visual Observations (`observation.images.top` + `observation.images.wrist` @ 640×480, 30 Hz)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    out_path = ASSETS_DIR / "hardware/camera_views_overview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"   Saved -> {out_path}")


def create_rollout_gif(run_dir: Path, out_gif: Path, title_text: str, step: int = 8, max_frame: int | None = None):
    top_mp4 = run_dir / "videos/observation.images.top/chunk-000/file-000.mp4"
    wrist_mp4 = run_dir / "videos/observation.images.wrist/chunk-000/file-000.mp4"

    top_all = decode_av1_mp4(top_mp4)
    wrist_all = decode_av1_mp4(wrist_mp4)
    n_total = min(len(top_all), len(wrist_all))
    if max_frame is not None:
        n_total = min(n_total, max_frame)

    indices = list(range(0, n_total, step))
    frames_pil = []
    target_w, target_h = 320, 240
    banner_h = 34

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx in indices:
        img_top = Image.fromarray(top_all[idx]).resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_wrist = Image.fromarray(wrist_all[idx]).resize((target_w, target_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (target_w * 2, target_h + banner_h), (24, 28, 36))
        canvas.paste(img_top, (0, banner_h))
        canvas.paste(img_wrist, (target_w, banner_h))

        draw = ImageDraw.Draw(canvas)
        t_sec = idx / 30.0
        header = f"{title_text}  |  t = {t_sec:4.1f}s (Frame {idx}/{n_total})"
        draw.text((12, 8), header, fill=(245, 246, 250), font=font)
        draw.text((10, banner_h + 6), "Top Camera", fill=(0, 230, 180), font=font)
        draw.text((target_w + 10, banner_h + 6), "Wrist Camera", fill=(255, 200, 60), font=font)

        frames_pil.append(canvas)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=frames_pil[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    size_mb = out_gif.stat().st_size / (1024 * 1024)
    print(f"   Saved GIF ({len(frames_pil)} frames, {size_mb:.2f} MB) -> {out_gif}")


def generate_all_gifs():
    print("[4/4] Generating 4 Real-Robot Closed-Loop Rollout GIFs from local/eval_*_smooth...")
    demos = [
        (
            LOCAL_EVAL_DIR / "eval_turbovla_smooth",
            ASSETS_DIR / "demos/rollout_turbovla_smooth.gif",
            "TurboVLA (Frozen-DINOv3 8k, Quintic C2 Smooth)",
            6,
            360,
        ),
        (
            LOCAL_EVAL_DIR / "eval_pi05_smooth",
            ASSETS_DIR / "demos/rollout_pi05_rtc.gif",
            "Pi0.5 (Full 20k, Flow-Matching RTC)",
            6,
            405,
        ),
        (
            LOCAL_EVAL_DIR / "eval_smolvla_smooth",
            ASSETS_DIR / "demos/rollout_smolvla_ensemble.gif",
            "SmolVLA (30k, Flow-Matching RTC)",
            7,
            481,
        ),
        (
            LOCAL_EVAL_DIR / "eval_act_smooth",
            ASSETS_DIR / "demos/rollout_act50_ensemble.gif",
            "ACT (50k, Temporal Ensemble lambda=0.01)",
            9,
            710,
        ),
    ]
    for r_dir, out_gif, title, step, max_f in demos:
        if r_dir.exists():
            create_rollout_gif(r_dir, out_gif, title, step=step, max_frame=max_f)
        else:
            print(f"   [Skip] {r_dir} not found")


if __name__ == "__main__":
    plot_turbovla_warmstart_vs_scratch()
    plot_temporal_smoothing_ablation()
    generate_camera_overview()
    generate_all_gifs()
    print("All visual assets generated successfully!")
