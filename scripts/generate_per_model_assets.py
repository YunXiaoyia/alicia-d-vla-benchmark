#!/usr/bin/env python3
"""Generate per-model ('分比' / Vertical Ablation) comparison plots and side-by-side rollout GIFs for all 5 policies:
- `act/`
- `smolvla/`
- `pi0_5/`
- `starvla/`
- `turbovla/`
"""

import glob
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
LOCAL_EVAL_DIR = WORKSPACE / "eval_runs/local"


def load_run_telemetry(run_name: str):
    pq_files = sorted(glob.glob(str(LOCAL_EVAL_DIR / run_name / "data/**/*.parquet"), recursive=True))
    if not pq_files:
        raise FileNotFoundError(f"No parquet found for {run_name}")
    table = pq.read_table(pq_files[0])
    act_full = np.stack(table["action"].to_pylist())
    st_full = np.stack(table["observation.state"].to_pylist())
    act_deg = act_full[:, :6]
    st_deg = st_full[:, :6]
    grip_cmd = act_full[:, 6]
    grip_st = st_full[:, 6]

    act_rad = act_deg * (np.pi / 180.0)
    vel_deg = np.diff(act_deg, axis=0)
    acc_deg = np.diff(vel_deg, axis=0)
    jerk_deg = np.diff(acc_deg, axis=0)
    l2_rad = np.linalg.norm(np.diff(act_rad, axis=0), axis=1)
    l2_deg = np.linalg.norm(vel_deg, axis=1)

    v0, v1 = vel_deg[:-1], vel_deg[1:]
    active = (np.abs(v0) > 0.02) & (np.abs(v1) > 0.02)
    flips = ((v0 * v1) < 0) & active
    sfr = float(flips.sum()) / max(int(active.sum()), 1) * 100.0

    return {
        "n_steps": len(act_deg),
        "act_deg": act_deg,
        "st_deg": st_deg,
        "grip_cmd": grip_cmd,
        "grip_st": grip_st,
        "vel_deg": vel_deg,
        "acc_deg": acc_deg,
        "jerk_deg": jerk_deg,
        "l2_rad": l2_rad,
        "l2_deg": l2_deg,
        "l2_rad_mean": float(np.mean(l2_rad)),
        "l2_rad_p95": float(np.percentile(l2_rad, 95)),
        "sfr": sfr,
        "mean_acc": float(np.mean(np.abs(acc_deg))),
        "mean_jerk": float(np.mean(np.abs(jerk_deg))),
        "max_jump": float(np.max(np.abs(vel_deg))),
    }


def plot_model_smooth_vs_no_smooth(
    model_title: str,
    run_no_smooth: str,
    run_smooth: str,
    smooth_label: str,
    out_png: Path,
):
    d_no = load_run_telemetry(run_no_smooth)
    d_sm = load_run_telemetry(run_smooth)

    fig = plt.figure(figsize=(20, 10.5), dpi=170)
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.26)

    t_no = np.arange(d_no["n_steps"]) / 30.0
    t_sm = np.arange(d_sm["n_steps"]) / 30.0

    joint_titles = [
        "Joint 1 (Base Yaw)",
        "Joint 2 (Shoulder Pitch)",
        "Joint 3 (Elbow Pitch)",
        "Joint 4 (Forearm Roll)",
        "Joint 5 (Wrist Pitch)",
        "Joint 6 (Wrist Roll)",
    ]
    panel_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for j_idx, (r_idx, c_idx) in enumerate(panel_positions):
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.plot(
            t_no,
            d_no["act_deg"][:, j_idx],
            color="#e17055",
            alpha=0.80,
            linewidth=1.4,
            linestyle="--",
            label="Unsmoothed Cmd",
        )
        ax.plot(
            t_sm,
            d_sm["act_deg"][:, j_idx],
            color="#00b894",
            alpha=0.95,
            linewidth=2.0,
            linestyle="-",
            label=f"Smoothed Cmd ({smooth_label})",
        )
        ax.plot(
            t_sm,
            d_sm["st_deg"][:, j_idx],
            color="#2d3436",
            alpha=0.55,
            linewidth=1.2,
            linestyle=":",
            label="Measured State (Smooth)",
        )
        j_jump_no = float(np.max(np.abs(d_no["vel_deg"][:, j_idx])))
        j_jump_sm = float(np.max(np.abs(d_sm["vel_deg"][:, j_idx])))
        ax.set_title(
            f"{joint_titles[j_idx]}\nMax Step Jump: {j_jump_no:.2f}° -> {j_jump_sm:.2f}°",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)", fontsize=9.5)
        ax.set_ylabel("Angle (°)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.45)
        if j_idx == 0:
            ax.legend(fontsize=8.0, loc="best", framealpha=0.92)

    # Panel (0, 3): Step-to-Step Action Variation ||Δa_t||_2 (rad)
    ax_vel = fig.add_subplot(gs[0, 3])
    w = min(300, len(d_no["l2_rad"]), len(d_sm["l2_rad"]))
    ax_vel.plot(
        np.arange(w) / 30.0,
        d_no["l2_rad"][:w],
        color="#d63031",
        alpha=0.80,
        linewidth=1.35,
        label=f"Unsmoothed (Mean: {d_no['l2_rad_mean']:.4f} rad)",
    )
    ax_vel.plot(
        np.arange(w) / 30.0,
        d_sm["l2_rad"][:w],
        color="#00b894",
        alpha=0.95,
        linewidth=1.9,
        label=f"Smoothed (Mean: {d_sm['l2_rad_mean']:.4f} rad)",
    )
    ax_vel.set_title("Step-to-Step Action Delta ||Δa_t||₂ (rad)", fontsize=10.8, fontweight="bold")
    ax_vel.set_xlabel("Time (s)", fontsize=9.5)
    ax_vel.set_ylabel("||Δa_t||₂ (rad)", fontsize=9.5)
    ax_vel.grid(True, linestyle="--", alpha=0.45)
    ax_vel.legend(fontsize=8.0, loc="upper right", framealpha=0.92)

    # Panel (1, 3): Normalized Smoothness Metrics Comparison Bar Chart
    ax_bar = fig.add_subplot(gs[1, 3])
    metric_labels = ["Mean ||Δa||₂\n(×100 rad)", "Mean Accel\n(°/step²)", "Mean Jerk\n(°/step³)", "Max Jump\n(÷10 °)"]
    vals_no = [
        d_no["l2_rad_mean"] * 100.0,
        d_no["mean_acc"],
        d_no["mean_jerk"],
        d_no["max_jump"] / 10.0,
    ]
    vals_sm = [
        d_sm["l2_rad_mean"] * 100.0,
        d_sm["mean_acc"],
        d_sm["mean_jerk"],
        d_sm["max_jump"] / 10.0,
    ]
    x = np.arange(len(metric_labels))
    width = 0.35
    b1 = ax_bar.bar(x - width / 2, vals_no, width, label="Unsmoothed", color="#e17055", edgecolor="#2d3436")
    b2 = ax_bar.bar(x + width / 2, vals_sm, width, label=smooth_label, color="#00b894", edgecolor="#2d3436")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=8.8, fontweight="bold")
    ax_bar.set_title(
        f"Smoothness Summary (SignFlip: {d_no['sfr']:.1f}% -> {d_sm['sfr']:.1f}%)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_bar.set_ylim(0, max(vals_no + vals_sm) * 1.25)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax_bar.legend(fontsize=8.0, loc="upper right")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(vals_no + vals_sm) * 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    plt.suptitle(
        f"{model_title} — Real-Robot Closed-Loop Smooth vs. Unsmoothed Trajectory Ablation (30 Hz on Alicia-D)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=170)
    plt.close()
    print(f"   Saved plot -> {out_png}")


def plot_smolvla_horizon_ablation(out_png: Path):
    runs = [
        ("n=15 (No Smooth)", "eval_smolvla_no_smooth", "#e17055"),
        ("n=20 (No Smooth)", "eval_smolvla_nk20", "#fdcb6e"),
        ("n=30 (No Smooth)", "eval_smolvla_nk30", "#0984e3"),
        ("n=15 + RTC (Smooth)", "eval_smolvla_smooth", "#00b894"),
    ]
    loaded = [(label, load_run_telemetry(r_name), col) for label, r_name, col in runs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=170)
    plt.subplots_adjust(wspace=0.25)

    ax1 = axes[0]
    for label, d, col in loaded:
        t = np.arange(d["n_steps"]) / 30.0
        ax1.plot(t, d["act_deg"][:, 1], label=label, color=col, linewidth=1.8)
    ax1.set_title("SmolVLA Joint 2 (Shoulder) Trajectory Across Horizons", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Joint 2 Angle (°)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(fontsize=8.5, loc="best")

    ax2 = axes[1]
    labels = [x[0] for x in loaded]
    l2_vals = [x[1]["l2_rad_mean"] for x in loaded]
    cols = [x[2] for x in loaded]
    bars = ax2.bar(np.arange(len(labels)), l2_vals, width=0.5, color=cols, edgecolor="#2d3436")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8.8, fontweight="bold", rotation=10)
    ax2.set_ylabel("Mean Step Delta ||Δa_t||₂ (rad)", fontsize=10)
    ax2.set_title("Step-to-Step Action Variation (Lower = Smoother)", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(l2_vals) * 1.25)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0003, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax3 = axes[2]
    jerk_vals = [x[1]["mean_jerk"] for x in loaded]
    bars3 = ax3.bar(np.arange(len(labels)), jerk_vals, width=0.5, color=cols, edgecolor="#2d3436")
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, fontsize=8.8, fontweight="bold", rotation=10)
    ax3.set_ylabel("Mean Joint Jerk (°/step³)", fontsize=10)
    ax3.set_title("3rd-Order Joint Jerk Across Execution Horizons", fontsize=11.5, fontweight="bold")
    ax3.set_ylim(0, max(jerk_vals) * 1.25)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("SmolVLA Action Execution Horizon (n=15, 20, 30) & Flow-Matching RTC Ablation on Real Hardware", fontsize=13.5, fontweight="bold", y=1.03)
    plt.savefig(out_png, bbox_inches="tight", dpi=170)
    plt.close()
    print(f"   Saved plot -> {out_png}")


def parse_starvla_log(log_path: Path):
    steps, losses = [], []
    pat = re.compile(r"Step\s+(\d+)(?:/\d+)?\s+\|\s+action_dit_loss:\s+([\d.eE+-]+)")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def plot_starvla_3way_freeze_ablation(out_png: Path):
    s_unf, l_unf = parse_starvla_log(BENCH_ROOT / "docs/train_starvla_unfrozen.log")
    s_vit, l_vit = parse_starvla_log(BENCH_ROOT / "docs/train_starvla_freeze_vit.log")
    s_qwen, l_qwen = parse_starvla_log(BENCH_ROOT / "docs/train_starvla_fixqwen.log")

    def smooth(y, w=15):
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=175)
    plt.subplots_adjust(wspace=0.25)

    # Panel 1: DiT Flow-Matching Loss Convergence
    ax1 = axes[0]
    w = 15
    ax1.plot(s_unf, l_unf, color="#d63031", alpha=0.18, linewidth=0.9)
    ax1.plot(s_vit, l_vit, color="#0984e3", alpha=0.18, linewidth=0.9)
    ax1.plot(s_qwen, l_qwen, color="#00b894", alpha=0.22, linewidth=0.9)

    ax1.plot(s_unf[w - 1 :], smooth(l_unf, w), color="#d63031", linestyle="--", linewidth=2.2, label=f"Run A: Unfrozen VLM (20k: {l_unf[-1]:.4f}, 0% Real)")
    ax1.plot(s_vit[w - 1 :], smooth(l_vit, w), color="#0984e3", linestyle="-.", linewidth=2.2, label=f"Run B: FreezeViT Only (10k: {l_vit[-1]:.4f}, Drift)")
    ax1.plot(s_qwen[w - 1 :], smooth(l_qwen, w), color="#00b894", linestyle="-", linewidth=2.6, label=f"Run C: FixQwen Full VLM Freeze (10k: {l_qwen[-1]:.4f}, 100% Real)")
    ax1.set_yscale("log")
    ax1.set_title("StarVLA Action DiT Flow Loss Convergence (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Steps (Batch Size = 16)", fontsize=10)
    ax1.set_ylabel("Flow-Matching MSE Loss (Log Scale)", fontsize=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.legend(fontsize=8.3, loc="upper right", framealpha=0.95)

    # Panel 2: Open-Loop 6-Joint MAE vs Real-Robot Generalization
    ax2 = axes[1]
    labels = ["Run A: Unfrozen VLM\n(2.26B Trainable\n20k Steps)", "Run B: FreezeViT\n(1.93B Trainable\n10k Steps)", "Run C: FixQwen\n(0.21B Trainable\n10k Steps)"]
    openloop_maes = [0.422, 0.562, 0.639]
    colors = ["#d63031", "#0984e3", "#00b894"]
    x = np.arange(len(labels))
    bars = ax2.bar(x, openloop_maes, width=0.45, color=colors, edgecolor="#2d3436", linewidth=1.2, alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Open-Loop 6-Joint MAE (°) on Training Ep0", fontsize=10)
    ax2.set_title("Open-Loop Memorization vs. Real-Robot Success", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.85)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    outcomes = [
        "0.422° Open-Loop\n0% Real (VLM Overfit)",
        "0.562° Open-Loop\nPartial (2B LLM Drift)",
        "0.639° Open-Loop\n100% Real Success!",
    ]
    for bar, txt in zip(bars, outcomes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, txt, ha="center", va="bottom", fontsize=8.3, fontweight="bold")

    # Panel 3: Training Time & Efficiency on RTX 4090
    ax3 = axes[2]
    wall_mins = [171.0, 74.0, 44.0]
    speeds = [1.95, 2.25, 3.79]
    bars3 = ax3.bar(x, wall_mins, width=0.45, color=colors, edgecolor="#2d3436", linewidth=1.2, alpha=0.88)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.8, fontweight="bold")
    ax3.set_ylabel("Total Training Wall-Clock Time (Minutes on RTX 4090)", fontsize=10)
    ax3.set_title("Training Efficiency & Throughput (RTX 4090)", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 215)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)
    anns3 = [
        f"171 min (1.95 it/s)\n18.4 GB VRAM",
        f"74 min (2.25 it/s)\n15.8 GB VRAM",
        f"44 min (3.79 it/s, 3.9× Faster)\n9.6 GB VRAM | Best!",
    ]
    for bar, txt in zip(bars3, anns3):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4.0, txt, ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.suptitle(
        "StarVLA (Qwen3.5-2B-VL + Flow-Matching DiT-B) — 3-Way VLM Freezing Ablation on Alicia-D",
        fontsize=13.5,
        fontweight="bold",
        y=1.03,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=175)
    plt.close()
    print(f"   Saved plot -> {out_png}")


def decode_av1_mp4(video_path: Path) -> list[np.ndarray]:
    container = av.open(str(video_path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def create_side_by_side_comparison_gif(
    run_no_smooth: str,
    run_smooth: str,
    model_label: str,
    smooth_method_label: str,
    out_gif: Path,
    step: int = 8,
    max_frames: int = 480,
):
    dir_no = LOCAL_EVAL_DIR / run_no_smooth
    dir_sm = LOCAL_EVAL_DIR / run_smooth

    top_no = decode_av1_mp4(dir_no / "videos/observation.images.top/chunk-000/file-000.mp4")
    wrist_no = decode_av1_mp4(dir_no / "videos/observation.images.wrist/chunk-000/file-000.mp4")
    top_sm = decode_av1_mp4(dir_sm / "videos/observation.images.top/chunk-000/file-000.mp4")
    wrist_sm = decode_av1_mp4(dir_sm / "videos/observation.images.wrist/chunk-000/file-000.mp4")

    n_no = min(len(top_no), len(wrist_no), max_frames)
    n_sm = min(len(top_sm), len(wrist_sm), max_frames)
    n_total = max(n_no, n_sm)

    indices = list(range(0, n_total, step))
    frames_pil = []
    cell_w, cell_h = 240, 180
    banner_h = 36

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for idx in indices:
        i_no = min(idx, n_no - 1)
        i_sm = min(idx, n_sm - 1)

        t_no_img = Image.fromarray(top_no[i_no]).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        w_no_img = Image.fromarray(wrist_no[i_no]).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        t_sm_img = Image.fromarray(top_sm[i_sm]).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        w_sm_img = Image.fromarray(wrist_sm[i_sm]).resize((cell_w, cell_h), Image.Resampling.BILINEAR)

        sep_w = 6
        total_w = cell_w * 4 + sep_w
        canvas = Image.new("RGB", (total_w, cell_h + banner_h), (24, 28, 36))
        canvas.paste(t_no_img, (0, banner_h))
        canvas.paste(w_no_img, (cell_w, banner_h))
        canvas.paste(t_sm_img, (cell_w * 2 + sep_w, banner_h))
        canvas.paste(w_sm_img, (cell_w * 3 + sep_w, banner_h))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([cell_w * 2, 0, cell_w * 2 + sep_w, cell_h + banner_h], fill=(255, 255, 255))

        t_sec = idx / 30.0
        draw.text((10, 9), f"[Unsmoothed] {model_label} (t={t_sec:4.1f}s)", fill=(255, 130, 110), font=font)
        draw.text(
            (cell_w * 2 + sep_w + 10, 9),
            f"[Smoothed: {smooth_method_label}] {model_label} (t={t_sec:4.1f}s)",
            fill=(85, 239, 196),
            font=font,
        )

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
    print(f"   Saved comparison GIF ({len(frames_pil)} frames, {size_mb:.2f} MB) -> {out_gif}")


def main():
    print("=== Generating Per-Model ('分比') Plots & Side-by-Side Comparison GIFs Across All 5 Models ===")

    # 1. ACT
    print("\n--- [1/5] ACT (act/) ---")
    plot_model_smooth_vs_no_smooth(
        "ACT (50k, ResNet-18 + CVAE)",
        "eval_act_no_smooth",
        "eval_act_smooth",
        "Temporal Ensemble (λ=0.01)",
        BENCH_ROOT / "act/act_smooth_vs_no_smooth.png",
    )
    create_side_by_side_comparison_gif(
        "eval_act_no_smooth",
        "eval_act_smooth",
        "ACT (50k)",
        "Temporal Ensemble λ=0.01",
        BENCH_ROOT / "act/rollout_comparison_act.gif",
        step=10,
        max_frames=700,
    )
    shutil.copy2(BENCH_ROOT / "policies/lerobot_adapters/train_act_4090.sh", BENCH_ROOT / "act/train_act_4090.sh")
    shutil.copy2(BENCH_ROOT / "scripts/eval_act_smooth.sh", BENCH_ROOT / "act/eval_act_smooth.sh")
    shutil.copy2(BENCH_ROOT / "scripts/eval_act_no_smooth.sh", BENCH_ROOT / "act/eval_act_no_smooth.sh")

    # 2. SmolVLA
    print("\n--- [2/5] SmolVLA (smolvla/) ---")
    plot_model_smooth_vs_no_smooth(
        "SmolVLA (30k, SmolVLM2-500M + Flow Matching)",
        "eval_smolvla_no_smooth",
        "eval_smolvla_smooth",
        "Flow-Matching RTC",
        BENCH_ROOT / "smolvla/smolvla_smooth_vs_no_smooth.png",
    )
    plot_smolvla_horizon_ablation(BENCH_ROOT / "smolvla/smolvla_horizon_ablation.png")
    create_side_by_side_comparison_gif(
        "eval_smolvla_no_smooth",
        "eval_smolvla_smooth",
        "SmolVLA (30k)",
        "Flow-Matching RTC",
        BENCH_ROOT / "smolvla/rollout_comparison_smolvla.gif",
        step=8,
        max_frames=481,
    )

    # 3. pi0.5
    print("\n--- [3/5] pi0.5 (pi0_5/) ---")
    plot_model_smooth_vs_no_smooth(
        "π0.5 (Full 20k, PaliGemma-3B + Action Expert Flow Matching)",
        "eval_pi05_no_smooth",
        "eval_pi05_smooth",
        "Flow-Matching RTC",
        BENCH_ROOT / "pi0_5/pi05_smooth_vs_no_smooth.png",
    )
    create_side_by_side_comparison_gif(
        "eval_pi05_no_smooth",
        "eval_pi05_smooth",
        "Pi0.5 (20k)",
        "Flow-Matching RTC",
        BENCH_ROOT / "pi0_5/rollout_comparison_pi05.gif",
        step=7,
        max_frames=448,
    )

    # 4. StarVLA
    print("\n--- [4/5] StarVLA (starvla/) ---")
    plot_starvla_3way_freeze_ablation(BENCH_ROOT / "starvla/starvla_3way_freeze_ablation.png")
    plot_model_smooth_vs_no_smooth(
        "StarVLA (FixQwen 10k, Frozen Qwen3.5-2B-VL + Flow-Matching DiT-B)",
        "eval_starvla_fixqwen_no_smooth",
        "eval_starvla_fixqwen_smooth",
        "Quintic C² Blender",
        BENCH_ROOT / "starvla/starvla_smooth_vs_no_smooth.png",
    )
    create_side_by_side_comparison_gif(
        "eval_starvla_fixqwen_no_smooth",
        "eval_starvla_fixqwen_smooth",
        "StarVLA (FixQwen 10k)",
        "Quintic C2 Blender",
        BENCH_ROOT / "starvla/rollout_comparison_starvla.gif",
        step=8,
        max_frames=499,
    )

    # 5. TurboVLA
    print("\n--- [5/5] TurboVLA (turbovla/) ---")
    plot_model_smooth_vs_no_smooth(
        "TurboVLA (Warm-Full 12.5k, Frozen Real DINOv3 + 1-NFE MeanFlow)",
        "eval_turbovla_warm_full_no_smooth",
        "eval_turbovla_warm_full_smooth",
        "Quintic C² Blender",
        BENCH_ROOT / "turbovla/turbovla_smooth_vs_no_smooth.png",
    )
    create_side_by_side_comparison_gif(
        "eval_turbovla_warm_full_no_smooth",
        "eval_turbovla_warm_full_smooth",
        "TurboVLA (Warm-Full 12.5k)",
        "Quintic C2 Blender",
        BENCH_ROOT / "turbovla/rollout_comparison_turbovla.gif",
        step=10,
        max_frames=710,
    )

    print("\nAll 5 per-model ('分比') assets generated successfully!")


if __name__ == "__main__":
    main()
