#!/usr/bin/env python3
"""Generate publication-grade cross-model ('横向对比 / 总比') benchmark figures and real-robot rollout GIFs
for alicia-d-vla-benchmark across all 5 policy families (selecting the best configuration of each policy):
1. `ACT (50k)`
2. `SmolVLA (30k)`
3. `π0.5 (Full 20k)`
4. `StarVLA (FixQwen 10k)`
5. `TurboVLA (Warm-Full 12.5k)`

Generates:
1. `assets/benchmarks/trajectory_tracking_all_models.png` — 5-Model Open-Loop Trajectory Tracking Comparison
2. `assets/benchmarks/turbovla_warmstart_vs_scratch.png` — TurboVLA Cross-Embodiment Transfer & Selective Warm-Start Ablation
3. `assets/benchmarks/temporal_smoothing_ablation.png` — 5-Model Real-Robot Closed-Loop Temporal Smoothing Ablation
4. `assets/hardware/camera_views_overview.png` — Synchronized Dual-Camera (`top` + `wrist`) across 4 task stages
5. `assets/demos/rollout_*.gif` — 5 Real-Robot Closed-Loop Rollout GIFs (`TurboVLA`, `StarVLA`, `π0.5`, `SmolVLA`, `ACT`)
"""

import glob
import os
import re
import shutil
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


def plot_trajectory_tracking_5models():
    print("[1/5] Generating 5-Model Trajectory Tracking Comparison (`trajectory_tracking_all_models.png`)...")
    cache_path = ASSETS_DIR / "benchmarks/openloop_5models_eval.npz"
    d = np.load(cache_path)
    gt_actions = d["gt_actions"]
    total_frames = len(gt_actions)
    time_axis = np.arange(total_frames) / 30.0

    # Best variant of each of the 5 policy families
    model_specs = [
        ("TurboVLA (Warm-Full)", d["turbovla_warm_full_actions"], 16.8, "#00b894", "-"),
        ("π0.5 (Full 20k)", d["pi05_actions"], 112.0, "#d63031", ":"),
        ("SmolVLA (30k)", d["smolvla_actions"], 56.3, "#e67e22", "-."),
        ("StarVLA (FixQwen)", d["starvla_fixqwen_actions"], 115.0, "#6c5ce7", "--"),
        ("ACT (50k)", d["act_actions"], 6.1, "#0984e3", (0, (3, 1, 1, 1))),
    ]

    model_results = {}
    for name, actions, lat, color, style in model_specs:
        errs = np.abs(actions - gt_actions)
        model_results[name] = {
            "actions": actions,
            "errs": errs,
            "j_mae": float(errs[:, :6].mean()),
            "g_mae": float(errs[:, 6].mean()),
            "lat": lat,
            "color": color,
            "style": style,
        }

    joint_names = [
        "Joint 1 (Base Yaw)",
        "Joint 2 (Shoulder Pitch)",
        "Joint 3 (Elbow Pitch)",
        "Joint 4 (Forearm Roll)",
        "Joint 5 (Wrist Pitch)",
        "Joint 6 (Wrist Roll)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), dpi=180)
    axes = axes.flatten()
    color_gt = "#2d3436"

    for i in range(6):
        ax = axes[i]
        ax.plot(
            time_axis,
            gt_actions[:, i],
            label="Ground Truth (Human)",
            color=color_gt,
            linewidth=2.5,
            linestyle="-",
            zorder=5,
        )
        for m_name, m_data in model_results.items():
            err_val = m_data["errs"][:, i].mean()
            ax.plot(
                time_axis,
                m_data["actions"][:, i],
                label=f"{m_name} ({err_val:.2f}°)",
                color=m_data["color"],
                linewidth=1.75,
                linestyle=m_data["style"],
            )
        ax.set_title(f"{joint_names[i]} (Degrees)", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9.8)
        ax.set_ylabel("Angle (°)", fontsize=9.8)
        ax.grid(True, linestyle="--", alpha=0.48)
        ax.legend(fontsize=7.3, loc="best", framealpha=0.92)

    # Panel 6: Gripper Opening
    ax_grip = axes[6]
    ax_grip.plot(time_axis, gt_actions[:, 6], label="Ground Truth", color=color_gt, linewidth=2.5, zorder=5)
    for m_name, m_data in model_results.items():
        ax_grip.plot(
            time_axis,
            m_data["actions"][:, 6],
            label=f"{m_name} (MAE {m_data['g_mae']:.1f})",
            color=m_data["color"],
            linewidth=1.75,
            linestyle=m_data["style"],
        )
    ax_grip.set_title("Gripper Opening [0=Closed, 1000=Open]", fontsize=11.5, fontweight="bold")
    ax_grip.set_xlabel("Time (s)", fontsize=9.8)
    ax_grip.set_ylabel("Opening Units", fontsize=9.8)
    ax_grip.grid(True, linestyle="--", alpha=0.48)
    ax_grip.legend(fontsize=7.3, loc="best", framealpha=0.92)

    # Panel 7: Overall 6-Joint MAE Bar Chart + RTX 4090 Chunk Latency Inset
    ax_bar = axes[7]
    short_names = ["TurboVLA\n(Warm-Full)", "π0.5\n(Full 20k)", "SmolVLA\n(30k)", "StarVLA\n(FixQwen)", "ACT\n(50k)"]
    keys = list(model_results.keys())
    x = np.arange(len(keys))
    j_maes = [model_results[k]["j_mae"] for k in keys]
    colors = [model_results[k]["color"] for k in keys]

    bars = ax_bar.bar(x, j_maes, color=colors, edgecolor="#2d3436", linewidth=1.1, alpha=0.88, width=0.55)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_names, fontsize=8.8, fontweight="bold")
    ax_bar.set_title("Overall 6-Joint MAE (°) — Lower is Better", fontsize=11.5, fontweight="bold")
    ax_bar.set_ylabel("Mean Absolute Error (°)", fontsize=9.8)
    ax_bar.set_ylim(0, max(j_maes) * 1.28)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.48)

    for bar in bars:
        h = bar.get_height()
        ax_bar.annotate(
            f"{h:.3f}°",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )

    latency_summary = "RTX 4090 Chunk Latency:\n" + "\n".join(
        [f"• {k.split(' (')[0]}: {model_results[k]['lat']:.1f} ms" for k in keys]
    )
    ax_bar.text(
        0.04,
        0.95,
        latency_summary,
        transform=ax_bar.transAxes,
        fontsize=8.2,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92),
    )

    plt.suptitle(
        "Cross-Model Benchmark of All 5 Policy Families on Alicia-D Wipes Pick-and-Place Trajectory\n"
        "TurboVLA (Warm-Full) vs. π0.5 vs. SmolVLA vs. StarVLA (FixQwen) vs. ACT | Frame 150 -> 300 (5.0s, Chunk Horizon=15)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = ASSETS_DIR / "benchmarks/trajectory_tracking_all_models.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    shutil.copy2(out_path, ASSETS_DIR / "benchmarks/trajectory_tracking_all_models_starvla.png")
    print(f"   Saved -> {out_path}")


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
    print("[2/5] Generating TurboVLA Cross-Embodiment Transfer & Selective Warm-Start Ablation plot...")
    log_a = BENCH_ROOT / "docs/train_wipes_a_warmstart.log"
    log_b = BENCH_ROOT / "docs/train_wipes_b_scratch.log"
    log_c = BENCH_ROOT / "docs/train_wipes_freeze_vit.log"
    log_d = BENCH_ROOT / "docs/train_wipes_warm_body.log"
    log_e = BENCH_ROOT / "docs/train_wipes_warm_full.log"

    s_a, l_a, g_a, _, sp_a = parse_turbovla_log(log_a)
    s_b, l_b, g_b, _, sp_b = parse_turbovla_log(log_b)
    s_c, l_c, g_c, _, sp_c = parse_turbovla_log(log_c)
    s_d, l_d, g_d, _, sp_d = parse_turbovla_log(log_d)
    s_e, l_e, g_e, _, sp_e = parse_turbovla_log(log_e)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), dpi=180)
    plt.subplots_adjust(wspace=0.25)

    def smooth(y, box_pts=7):
        if len(y) < box_pts:
            return y
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode="valid")

    w = 7
    ax1 = axes[0]
    ax1.plot(s_b[w - 1 :], smooth(l_b, w), color="#d63031", linewidth=2.0, linestyle="--", label=f"Run B: Scratch Unfrozen (20k: {l_b[-1]:.4f}, 0% Real)")
    ax1.plot(s_a[w - 1 :], smooth(l_a, w), color="#e67e22", linewidth=2.0, linestyle=":", label=f"Run A: Sim-ViT Warm-Start (20k: {l_a[-1]:.4f}, 0% Real)")
    ax1.plot(s_c[w - 1 :], smooth(l_c, w), color="#0984e3", linewidth=2.2, linestyle="-.", label=f"Run C: Frozen-DINOv3 Warm-Full (8k: {l_c[-1]:.4f}, 100% Real)")
    ax1.plot(s_d[w - 1 :], smooth(l_d, w), color="#6c5ce7", linewidth=2.2, linestyle="--", label=f"Run D: Frozen-DINOv3 Warm-Body (12.5k: {l_d[-1]:.4f}, Partial)")
    ax1.plot(s_e[w - 1 :], smooth(l_e, w), color="#00b894", linewidth=2.8, linestyle="-", label=f"Run E: Frozen-DINOv3 Warm-Full (12.5k: {l_e[-1]:.4f}, 100% Best)")
    ax1.set_yscale("log")
    ax1.set_title("TurboVLA Action L1 Loss Convergence (Log Scale)", fontsize=12.2, fontweight="bold")
    ax1.set_xlabel("Training Steps (Effective Batch Size = 64)", fontsize=10.2)
    ax1.set_ylabel("L1 Action Loss (Log Scale)", fontsize=10.2)
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.legend(fontsize=7.8, loc="upper right", framealpha=0.95)

    # Panel 2: Open-Loop 6-Joint MAE vs Real-Robot Transfer Outcome
    ax2 = axes[1]
    labels = [
        "Run B\nScratch\n(20k)",
        "Run A\nSim-ViT\n(20k)",
        "Run C\nFrozen-ViT\n(8k)",
        "Run D\nWarm-Body\n(12.5k)",
        "Run E\nWarm-Full\n(12.5k)",
    ]
    openloop_maes = [0.145, 0.088, 0.606, 0.171, 0.166]
    bar_colors = ["#d63031", "#e67e22", "#0984e3", "#6c5ce7", "#00b894"]
    x = np.arange(len(labels))
    bars2 = ax2.bar(x, openloop_maes, width=0.52, color=bar_colors, edgecolor="#2d3436", linewidth=1.1, alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.2, fontweight="bold")
    ax2.set_ylabel("Open-Loop 6-Joint MAE (°) on Training Ep0", fontsize=10.2)
    ax2.set_title("Open-Loop Precision & Real-Robot Transfer Outcome", fontsize=12.2, fontweight="bold")
    ax2.set_ylim(0, 0.78)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)

    outcomes = [
        "0.145°\n0% (Overfit)",
        "0.088°\n0% (Sim Drift)",
        "0.606°\n100% Real",
        "0.171°\nRand Head",
        "0.166°\n100% Best!",
    ]
    for bar, ann in zip(bars2, outcomes):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            ann,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#2d3436",
        )

    # Panel 3: Training Wall-Clock Time Comparison
    ax3 = axes[2]
    wall_mins = [162.8, 162.4, 23.0, 36.2, 36.3]
    speeds_val = [sp_b.mean(), sp_a.mean(), sp_c.mean(), sp_d.mean(), sp_e.mean()]
    bars3 = ax3.bar(x, wall_mins, width=0.52, color=bar_colors, edgecolor="#2d3436", linewidth=1.1, alpha=0.88)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.2, fontweight="bold")
    ax3.set_ylabel("Total Training Wall-Clock Time (Minutes on RTX 4090)", fontsize=10.2)
    ax3.set_title("Training Wall-Clock Time & Throughput (RTX 4090)", fontsize=12.2, fontweight="bold")
    ax3.set_ylim(0, 210)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)

    annotations = [
        f"163m\n({speeds_val[0]:.2f} it/s)",
        f"162m\n({speeds_val[1]:.2f} it/s)",
        f"23m (7.1×)\n({speeds_val[2]:.2f} it/s)",
        f"36m (4.5×)\n({speeds_val[3]:.2f} it/s)",
        f"36m (4.5×)\n({speeds_val[4]:.2f} it/s)",
    ]
    for bar, ann in zip(bars3, annotations):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4.0,
            ann,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#2d3436",
        )

    plt.suptitle(
        "TurboVLA Cross-Embodiment Transfer & Selective Warm-Start Ablation (LIBERO Simulation -> Alicia-D Real Hardware)",
        fontsize=13.8,
        fontweight="bold",
        y=1.03,
    )
    out_path = ASSETS_DIR / "benchmarks/turbovla_warmstart_vs_scratch.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    shutil.copy2(out_path, BENCH_ROOT / "turbovla/turbovla_3way_transfer_ablation.png")
    print(f"   Saved -> {out_path} (and copied to turbovla/turbovla_3way_transfer_ablation.png)")


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
    print("[3/5] Generating 5-Model Real-Robot Temporal Smoothing Ablation plot...")
    models_cfg = [
        ("ACT (50k)", "eval_act_no_smooth", "eval_act_smooth", "Temporal Ensemble (λ=0.01)"),
        ("SmolVLA (30k)", "eval_smolvla_no_smooth", "eval_smolvla_smooth", "Flow-Matching RTC"),
        ("π0.5 (Full 20k)", "eval_pi05_no_smooth", "eval_pi05_smooth", "Flow-Matching RTC"),
        ("StarVLA (FixQwen 10k)", "eval_starvla_fixqwen_no_smooth", "eval_starvla_fixqwen_smooth", "Quintic C² Blender"),
        ("TurboVLA (Warm-Full 12.5k)", "eval_turbovla_warm_full_no_smooth", "eval_turbovla_warm_full_smooth", "Quintic C² Blender"),
    ]

    data = {}
    for label, r_no, r_sm, method in models_cfg:
        data[label] = {
            "no": load_real_eval_run(r_no),
            "sm": load_real_eval_run(r_sm),
            "method": method,
        }

    fig = plt.figure(figsize=(23, 10.2), dpi=180)
    gs_top = fig.add_gridspec(1, 5, top=0.92, bottom=0.55, hspace=0.35, wspace=0.25)
    gs_bot = fig.add_gridspec(1, 4, top=0.43, bottom=0.06, hspace=0.35, wspace=0.26)

    # Top Row: 5 Models Step-to-Step Velocity / Trajectory Smoothness
    for idx, (label, _, _, method) in enumerate(models_cfg):
        ax = fig.add_subplot(gs_top[0, idx])
        d_no = data[label]["no"]
        d_sm = data[label]["sm"]

        w = min(240, len(d_no["l2_rad"]), len(d_sm["l2_rad"]))
        t_axis = np.arange(w) / 30.0
        ax.plot(
            t_axis,
            d_no["l2_rad"][:w],
            color="#d63031",
            alpha=0.78,
            linewidth=1.3,
            label=f"No-Smooth ({d_no['l2_rad_mean']:.4f})",
        )
        ax.plot(
            t_axis,
            d_sm["l2_rad"][:w],
            color="#00b894",
            alpha=0.95,
            linewidth=1.8,
            label=f"Smooth ({d_sm['l2_rad_mean']:.4f})",
        )
        ax.set_title(f"{label}\n[{method}]", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9.2)
        ax.set_ylabel("Step Delta ||Δa_t||₂ (rad)", fontsize=9.2)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=7.6, loc="upper right", framealpha=0.92)

    # Bottom Row: 4 Quantitative Comparison Bar Charts across All 5 Models
    model_names = [m[0].split(" (")[0] for m in models_cfg]
    x = np.arange(len(model_names))
    width = 0.36

    metrics_panels = [
        ("l2_rad_mean", "Mean Step Delta ||Δa_t||₂ (rad)", "Step-to-Step Action Variation (Lower = Smoother)", "{:.4f}"),
        ("mean_jerk", "Mean Joint Jerk (°/step³)", "3rd-Order Joint Jerk (Lower = Less Vibration)", "{:.3f}"),
        ("sfr", "Velocity Sign-Flip Rate (%)", "High-Freq Velocity Reversal Rate (%)", "{:.1f}%"),
        ("max_jump", "Max Single-Step Jump (°)", "Worst Chunk-Boundary Step Jump (°)", "{:.2f}°"),
    ]

    for p_idx, (m_key, ylabel, title, fmt) in enumerate(metrics_panels):
        ax = fig.add_subplot(gs_bot[0, p_idx])
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
            label="Smoothed (Ensemble / RTC / Quintic C²)",
            color="#00b894",
            edgecolor="#2d3436",
            linewidth=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9.0, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9.2)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, max(vals_no + vals_sm) * 1.26)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        if p_idx == 0:
            ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(vals_no + vals_sm) * 0.02,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )

    plt.suptitle(
        "Real-Robot Closed-Loop Temporal Smoothing Ablation Across All 5 Policy Architectures (30 Hz on Alicia-D)",
        fontsize=14.2,
        fontweight="bold",
        y=0.99,
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
    print("[4/5] Generating Dual-Camera Overview figure from real-robot TurboVLA Warm-Full rollout...")
    run_dir = LOCAL_EVAL_DIR / "eval_turbovla_warm_full_smooth"
    top_frames = decode_av1_mp4(run_dir / "videos/observation.images.top/chunk-000/file-000.mp4")
    wrist_frames = decode_av1_mp4(run_dir / "videos/observation.images.wrist/chunk-000/file-000.mp4")

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
    print("[5/5] Generating 5 Real-Robot Closed-Loop Rollout GIFs (Best Variant of Each Policy)...")
    demos = [
        (
            LOCAL_EVAL_DIR / "eval_turbovla_warm_full_smooth",
            ASSETS_DIR / "demos/rollout_turbovla_smooth.gif",
            "TurboVLA (Warm-Full 12.5k, Quintic C2 Smooth)",
            7,
            420,
        ),
        (
            LOCAL_EVAL_DIR / "eval_starvla_fixqwen_smooth",
            ASSETS_DIR / "demos/rollout_starvla_smooth.gif",
            "StarVLA (FixQwen 10k, Quintic C2 Smooth)",
            7,
            470,
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
            700,
        ),
    ]
    for r_dir, out_gif, title, step, max_f in demos:
        if r_dir.exists():
            create_rollout_gif(r_dir, out_gif, title, step=step, max_frame=max_f)
        else:
            print(f"   [Skip] {r_dir} not found")


if __name__ == "__main__":
    plot_trajectory_tracking_5models()
    plot_turbovla_warmstart_vs_scratch()
    plot_temporal_smoothing_ablation()
    generate_camera_overview()
    generate_all_gifs()
    print("All cross-model ('横向对比 / 总比') visual assets generated successfully!")
