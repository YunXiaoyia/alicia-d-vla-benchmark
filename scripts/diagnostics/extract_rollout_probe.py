#!/usr/bin/env python3
"""Extract RGB observations and executed commands from a LeRobot v3 rollout.

The future commands are a policy rollout reference, not expert ground truth.
This CPU-only utility does not load a model or connect to robot hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd


def read_selected_rgb(path: Path, timestamps: np.ndarray, fps: float):
    wanted = np.rint(timestamps * fps).astype(np.int64)
    images = {}
    actual_times = {}
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            timestamp = float(frame.time)
            index = int(round(timestamp * fps))
            if index in wanted and index not in images:
                images[index] = frame.to_ndarray(format="rgb24")
                actual_times[index] = timestamp
            if index >= int(wanted.max()):
                break
    missing = sorted(set(wanted.tolist()) - images.keys())
    if missing:
        raise ValueError(f"Missing video frames in {path}: {missing}")
    decoded_times = np.asarray([actual_times[int(i)] for i in wanted])
    if np.max(np.abs(decoded_times - timestamps)) > 0.5 / fps + 1e-6:
        raise ValueError(f"Video timestamps do not align with observations: {path}")
    return np.stack([images[int(i)] for i in wanted]), decoded_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", default="0,25,50,70,85,100,110,118,125,140,155,170,190,200,206,220,250,300")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--action-stats", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("*/*.parquet"))],
        ignore_index=True,
    )
    episode = episodes.loc[episodes.episode_index == args.episode].iloc[0]
    data = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))],
        ignore_index=True,
    )
    data = data.loc[data.episode_index == args.episode].sort_values("frame_index").reset_index(drop=True)
    selected = np.asarray(sorted(set(int(f) for f in args.frames.split(","))), dtype=np.int64)
    positions = data.set_index("frame_index").index.get_indexer(selected)
    if (positions < 0).any() or args.horizon < 1:
        raise ValueError("Requested frames must exist and horizon must be positive")
    states = np.stack(data["observation.state"]).astype(np.float32)
    actions = np.stack(data.action).astype(np.float32)
    future_indices = positions[:, None] + np.arange(args.horizon)[None, :]
    valid = future_indices < len(data)
    future = actions[np.minimum(future_indices, len(data) - 1)]
    timestamps = data.timestamp.to_numpy(dtype=np.float64)[positions]
    arrays = {
        "frame_index": selected,
        "episode_index": np.full(len(selected), args.episode, dtype=np.int64),
        "timestamp": timestamps,
        "state": states[positions],
        "future_actions": future,
        "future_action_mask": valid,
        "reference_kind": np.asarray("executed_policy_commands_not_expert_ground_truth"),
    }
    for camera in ("top", "wrist"):
        key = f"observation.images.{camera}"
        prefix = f"videos/{key}"
        path = root / info["video_path"].format(
            video_key=key,
            chunk_index=int(episode[f"{prefix}/chunk_index"]),
            file_index=int(episode[f"{prefix}/file_index"]),
        )
        video_times = timestamps + float(episode[f"{prefix}/from_timestamp"])
        arrays[camera], arrays[f"{camera}_video_timestamp"] = read_selected_rgb(path, video_times, fps)

    grip = actions[:, 6]
    transitions = np.flatnonzero((grip[1:] < 500) != (grip[:-1] < 500)) + 1
    measured_grip = states[:, 6]
    measured_transitions = np.flatnonzero((measured_grip[1:] < 500) != (measured_grip[:-1] < 500)) + 1
    summary = {
        "source_dataset": str(root),
        "episode": args.episode,
        "total_frames": len(data),
        "fps": fps,
        "sample_frames": selected.tolist(),
        "reference_kind": str(arrays["reference_kind"]),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "state_min": states.min(axis=0).tolist(),
        "state_max": states.max(axis=0).tolist(),
        "gripper_command_500_crossings": [
            {
                "frame_index": int(data.frame_index.iloc[i]),
                "direction": "closing" if grip[i] < 500 else "opening",
                "action": actions[i].tolist(),
                "state": states[i].tolist(),
            }
            for i in transitions
        ],
        "gripper_measured_500_crossings": [
            {
                "frame_index": int(data.frame_index.iloc[i]),
                "direction": "closing" if measured_grip[i] < 500 else "opening",
                "state": states[i].tolist(),
            }
            for i in measured_transitions
        ],
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    if args.action_stats:
        limits = json.loads(args.action_stats.read_text())["wipes"]["action"]
        lo, hi = np.asarray(limits["min"]), np.asarray(limits["max"])
        summary["action_limits"] = {"min": lo.tolist(), "max": hi.tolist()}
        summary["action_outside_limits_count_per_joint"] = ((actions < lo) | (actions > hi)).sum(axis=0).tolist()
        summary["state_outside_action_limits_count_per_joint"] = ((states < lo) | (states > hi)).sum(axis=0).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
