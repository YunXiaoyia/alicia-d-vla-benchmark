"""LeRobot Wipes Dataset adapter for TurboVLA."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotWipesDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "/home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01",
        stats_path: str = "/home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json",
        chunk_size: int = 30,
        expected_image_size: int = 256,
        split: str = "train",
        split_ratio: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.expected_image_size = expected_image_size
        self.dataset_dir = dataset_dir

        self.lerobot_ds = LeRobotDataset("wipes-to-green-plate-01", root=dataset_dir)

        # Load all parquet data into memory for fast action & state indexing (only ~700 KB)
        files = sorted(glob.glob(os.path.join(dataset_dir, "data/*/*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {dataset_dir}/data/*/*.parquet")
        dfs = [pd.read_parquet(f) for f in files]
        full_df = pd.concat(dfs, ignore_index=True)
        self.all_actions = np.stack(full_df["action"].values).astype(np.float32)
        self.all_states = np.stack(full_df["observation.state"].values).astype(np.float32)

        # Load stats for normalization
        stats_data = json.loads(Path(stats_path).read_text(encoding="utf-8"))["wipes"]
        self.action_min = np.array(stats_data["action"]["min"], dtype=np.float32)
        self.action_max = np.array(stats_data["action"]["max"], dtype=np.float32)
        self.proprio_mean = np.array(stats_data["proprio"]["mean"], dtype=np.float32)
        self.proprio_std = np.array(stats_data["proprio"]["std"], dtype=np.float32)

        # Build episode lookup: mapping from frame_index to episode end
        ep_meta = self.lerobot_ds.meta.episodes
        num_episodes = len(ep_meta)
        split_idx = int(num_episodes * split_ratio)
        if split == "train":
            ep_range = range(0, split_idx)
        else:
            ep_range = range(split_idx, num_episodes)

        self.valid_indices: List[int] = []
        self.frame_to_ep_to: Dict[int, int] = {}
        for ep_i in ep_range:
            row = ep_meta[ep_i]
            frm = int(row["dataset_from_index"])
            to = int(row["dataset_to_index"])
            for f in range(frm, to):
                self.valid_indices.append(f)
                self.frame_to_ep_to[f] = to

        self.instruction = "Pick up the pack of cleaning wipes and place it on the green plate"
        # Standard ImageNet / DINOv3 normalization
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _process_image(self, img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        # img_tensor: [3, H, W] in [0, 1]
        img_resized = TF.resize(img_tensor, [self.expected_image_size, self.expected_image_size], antialias=True)
        img_norm = (img_resized - self.img_mean) / self.img_std
        return {"dinov3": img_norm}

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        ep_to = self.frame_to_ep_to[real_idx]

        # 1. Fetch images from LeRobot dataset
        item = self.lerobot_ds[real_idx]
        top_img = item["observation.images.top"]
        wrist_img = item["observation.images.wrist"]

        img1 = self._process_image(top_img)
        img2 = self._process_image(wrist_img)

        # 2. State: pad 7-dim to 8-dim with 0.0 for TurboVLA pretrain compatibility, then normalize
        raw_state = self.all_states[real_idx]
        state_8d = np.append(raw_state, 0.0).astype(np.float32)
        norm_state = (state_8d - self.proprio_mean) / (self.proprio_std + 1e-8)
        state_tensor = torch.tensor(norm_state, dtype=torch.float32)

        # 3. Future action chunk
        valid_actions = self.all_actions[real_idx : min(real_idx + self.chunk_size, ep_to)]
        valid_len = len(valid_actions)
        padding_len = self.chunk_size - valid_len

        if padding_len > 0:
            last_action = valid_actions[-1:]
            padded_actions = np.repeat(last_action, padding_len, axis=0)
            full_actions = np.concatenate([valid_actions, padded_actions], axis=0)
        else:
            full_actions = valid_actions

        # Normalize action to [-1, 1]
        norm_actions = 2.0 * (full_actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1.0
        norm_actions = np.clip(norm_actions, -1.0, 1.0)
        action_chunk = torch.tensor(norm_actions, dtype=torch.float32)

        action_chunk_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        action_chunk_mask[:valid_len] = 1.0

        return (img1, img2), self.instruction, state_tensor, action_chunk, action_chunk_mask


def wipes_collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("empty batch cannot be collated")

    dino_img1_list = []
    dino_img2_list = []
    instructions = []
    states = []
    action_chunks = []
    action_chunk_masks = []

    for (img1, img2), instruction, state, action_chunk, action_chunk_mask in batch:
        dino_img1_list.append(img1["dinov3"])
        dino_img2_list.append(img2["dinov3"])
        instructions.append(instruction)
        states.append(state)
        action_chunks.append(action_chunk)
        action_chunk_masks.append(action_chunk_mask)

    dino_img1 = torch.stack(dino_img1_list, dim=0)
    dino_img2 = torch.stack(dino_img2_list, dim=0)

    samples = {"dinov3": torch.stack([dino_img1, dino_img2], dim=1)}
    states = torch.stack(states, dim=0)
    action_chunks = torch.stack(action_chunks, dim=0)
    action_chunk_masks = torch.stack(action_chunk_masks, dim=0)

    return samples, instructions, states, action_chunks, action_chunk_masks