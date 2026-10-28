#!/usr/bin/env python3
"""Evaluate a trained TurboVLA checkpoint (.pth) on the Alicia-D robot arm via LeRobot.

Aligns 100% with `TurboVLA/experiments/wipes/train_wipes.py` and `wipes_dataset.py`:
- Dual camera inputs (`observation.images.top`, `observation.images.wrist`) resized to 256x256
  with bilinear antialiasing and ImageNet mean/std normalization (NO 180-deg LIBERO flip).
- 7D joint+gripper state (`[joint1..6 (deg), gripper (0~1000)]`) padded with `0.0` to 8D and
  Z-score normalized using `wipes_stats.json["wipes"]["proprio"]`.
- 30-step 7D action chunk unnormalized from `[-1, 1]` using `wipes_stats.json["wipes"]["action"]`
  into absolute joint degrees (`joint1..6.pos`) and gripper opening (`gripper.pos` in `[0, 1000]`).

Example usage (Warm-start checkpoint):
```shell
conda activate lerobot
cd /home/yunyi/work/vla/lerobot
python examples/alicia/eval_alicia_turbovla.py \
    --ckpt_path=/home/yunyi/work/vla/weights/turbovla_wipes/turbovla_wipes_warmstart_step_20000.pth \
    --robot.type=alicia_d_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{
        top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30},
        wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}
    }' \
    --n_action_steps=15 \
    --fps=30 \
    --duration=120 \
    --num_episodes=5
```
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Add TurboVLA repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
TURBOVLA_ROOT = REPO_ROOT / "TurboVLA"
if str(TURBOVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(TURBOVLA_ROOT))

from turbovla.models.turbovla import build_turbovla  # noqa: E402

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from lerobot.configs import parser  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import hw_to_dataset_features  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import (  # noqa: F401, E402
    RobotConfig,
    alicia_d_follower,
    bi_alicia_d_follower,
    make_robot_from_config,
)
from lerobot.scripts.lerobot_record import record_loop  # noqa: E402
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR  # noqa: E402
from lerobot.utils.control_utils import init_keyboard_listener, is_headless  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WIPES_TASK = "Pick up the pack of cleaning wipes and place it on the green plate"


class _IdentityPolicyProcessor:
    """No-op processor compatible with LeRobot's `record_loop` / `predict_action`."""

    def reset(self) -> None:
        pass

    def __call__(self, data: Any) -> Any:
        return data


class TurboVLAAliciaPolicy(nn.Module):
    """LeRobot-compatible policy wrapper around a fine-tuned TurboVLA `.pth` checkpoint."""

    def __init__(
        self,
        ckpt_path: str | Path,
        stats_path: str | Path,
        dinov3_path: str | Path,
        bert_path: str | Path,
        device: str = "cuda",
        chunk_size: int = 30,
        n_action_steps: int = 15,
        temporal_ensemble_coeff: float | None = None,
        use_ema: bool = True,
        expected_image_size: int = 256,
        default_task: str = DEFAULT_WIPES_TASK,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = int(chunk_size)
        self.n_action_steps = max(1, min(int(n_action_steps), self.chunk_size))
        self.temporal_ensemble_coeff = (
            None if temporal_ensemble_coeff in (None, 0.0) else float(temporal_ensemble_coeff)
        )
        self.expected_image_size = int(expected_image_size)
        self.default_task = str(default_task)

        # Required by LeRobot's `predict_action` / `record_loop`
        self.config = SimpleNamespace(
            device=str(self.device),
            use_amp=False,  # Autocast (bfloat16) is handled explicitly inside _predict_action_chunk
            type="turbovla",
        )

        # Load normalization statistics from wipes_stats.json
        stats_raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        stats_data = stats_raw["wipes"] if "wipes" in stats_raw else stats_raw
        self.register_buffer(
            "action_min",
            torch.tensor(stats_data["action"]["min"], dtype=torch.float32, device=self.device).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "action_max",
            torch.tensor(stats_data["action"]["max"], dtype=torch.float32, device=self.device).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "proprio_mean",
            torch.tensor(stats_data["proprio"]["mean"], dtype=torch.float32, device=self.device).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "proprio_std",
            torch.tensor(stats_data["proprio"]["std"], dtype=torch.float32, device=self.device).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )

        # Build TurboVLA architecture matching `train_wipes.py`
        model_args = SimpleNamespace(
            dinov3_path=str(dinov3_path),
            bert_path=str(bert_path),
            hidden_dim=256,
            nheads=8,
            dim_feedforward=2048,
            max_text_len=256,
            text_padding_length=21,
            text_padding_length_by_instruction={},
            vla_feature_enhancer_layers=6,
            enhancer_inner_dim=1024,
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.1,
            action_dim=7,
            chunk_size=self.chunk_size,
            state_dim=8,
            num_state_tokens=2,
            local_files_only=True,
            freeze_vision_encoder=True,
            freeze_text_encoder=True,
            dinov3_precision="bf16_autocast",
            num_views=2,
            image_size=self.expected_image_size,
            position_embedding="view",
            encode_views_separately=True,
            padding_strategy="key_padding_mask",
        )
        logger.info("Building TurboVLA model...")
        self.model = build_turbovla(model_args)

        # Load checkpoint (.pth)
        logger.info(f"Loading TurboVLA checkpoint from {ckpt_path} (use_ema={use_ema})...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if use_ema and "ema_model_state_dict" in ckpt:
                state_dict = ckpt["ema_model_state_dict"]
                ckpt_source = "ema_model_state_dict"
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                ckpt_source = "model_state_dict"
            else:
                state_dict = ckpt
                ckpt_source = "raw_dict"
            global_step = ckpt.get("global_step", "unknown")
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

        cleaned_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            cleaned_sd[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned_sd, strict=False)
        if missing or unexpected:
            logger.warning(f"Checkpoint load non-strict keys: missing={missing}, unexpected={unexpected}")
        else:
            logger.info(
                f"Successfully loaded {len(cleaned_sd)} tensors from '{ckpt_source}' (step={global_step}) with strict match!"
            )

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.n_action_steps)
        self._ensemble_chunks: deque[torch.Tensor] = deque(maxlen=self.chunk_size)

    def reset(self) -> None:
        """Reset action queue / temporal ensemble buffer at the start of each episode."""
        self._action_queue.clear()
        self._ensemble_chunks.clear()

    def _preprocess_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Resize `[B, 3, H, W]` image in `[0, 1]` to `256x256` and normalize with ImageNet stats."""
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)
        img_resized = TF.resize(
            img_tensor,
            [self.expected_image_size, self.expected_image_size],
            antialias=True,
        )
        return (img_resized - self.img_mean) / self.img_std

    @torch.inference_mode()
    def _predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run one forward pass and return unnormalized action chunk `[1, chunk_size, 7]`."""
        if "observation.images.top" not in batch or "observation.images.wrist" not in batch:
            avail = list(batch.keys())
            raise KeyError(
                f"Expected 'observation.images.top' and 'observation.images.wrist' in observation, got {avail}"
            )

        dino_top = self._preprocess_image(batch["observation.images.top"])
        dino_wrist = self._preprocess_image(batch["observation.images.wrist"])
        samples = {"dinov3": torch.stack([dino_top, dino_wrist], dim=1)}  # [B, 2, 3, 256, 256]

        # State: [B, 7] -> pad 0.0 -> [B, 8] -> Z-score normalize
        raw_state = batch["observation.state"].to(device=self.device, dtype=torch.float32)
        if raw_state.ndim == 1:
            raw_state = raw_state.unsqueeze(0)
        if raw_state.shape[-1] == 7:
            state_8d = F.pad(raw_state, (0, 1), value=0.0)
        elif raw_state.shape[-1] == 8:
            state_8d = raw_state
        else:
            raise ValueError(f"Expected 7D or 8D observation.state, got shape {tuple(raw_state.shape)}")
        norm_state = (state_8d - self.proprio_mean) / (self.proprio_std + 1e-8)

        task = batch.get("task", self.default_task)
        if isinstance(task, (list, tuple)):
            task = task[0] if task else self.default_task
        if not task:
            task = self.default_task

        use_bf16 = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred_norm = self.model([str(task)], samples, norm_state)  # [1, 30, 7]

        pred_clipped = torch.clamp(pred_norm.float(), -1.0, 1.0)
        raw_actions = 0.5 * (pred_clipped + 1.0) * (self.action_max - self.action_min + 1e-8) + self.action_min
        # Clamp gripper command (index 6) to valid [0, 1000] range
        raw_actions[..., 6] = torch.clamp(raw_actions[..., 6], 0.0, 1000.0)
        return raw_actions

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return a single-step action `[1, 7]` for `record_loop`."""
        if self.temporal_ensemble_coeff is not None:
            chunk = self._predict_action_chunk(batch)[0]  # [chunk_size, 7]
            self._ensemble_chunks.append(chunk)
            # Oldest chunk is at index 0 (predicted k-1 steps ago -> take step index k-1)
            k = len(self._ensemble_chunks)
            candidates = torch.stack(
                [self._ensemble_chunks[i][k - 1 - i] for i in range(k)],
                dim=0,
            )  # [k, 7], ordered from oldest to newest
            # Exponential weights: w = exp(-coeff * age), where age = k - 1 - i
            ages = torch.arange(k - 1, -1, -1, dtype=torch.float32, device=self.device)
            weights = torch.exp(-self.temporal_ensemble_coeff * ages)
            weights = (weights / weights.sum()).unsqueeze(-1)
            action = (candidates * weights).sum(dim=0, keepdim=True)  # [1, 7]
            return action

        if len(self._action_queue) == 0:
            chunk = self._predict_action_chunk(batch)  # [1, chunk_size, 7]
            for step_idx in range(self.n_action_steps):
                self._action_queue.append(chunk[:, step_idx, :])

        return self._action_queue.popleft()


@dataclass
class TurboVLAEvalConfig:
    """Configuration for evaluating TurboVLA on the Alicia-D robot arm."""

    # TurboVLA checkpoint & model paths
    ckpt_path: str = field(
        default=str(REPO_ROOT / "weights/turbovla_wipes/turbovla_wipes_warmstart_step_20000.pth"),
        metadata={"help": "Path to fine-tuned TurboVLA .pth checkpoint (warmstart or scratch)"},
    )
    stats_path: str = field(
        default=str(REPO_ROOT / "weights/turbovla_wipes/wipes_stats.json"),
        metadata={"help": "Path to wipes_stats.json containing proprio and action normalization stats"},
    )
    dinov3_path: str = field(
        default=str(REPO_ROOT / "models/dinov3-vitb16-pretrain-lvd1689m"),
        metadata={"help": "Local directory for DINOv3 backbone"},
    )
    bert_path: str = field(
        default=str(REPO_ROOT / "models/bert-base-uncased"),
        metadata={"help": "Local directory for BERT-base-uncased backbone"},
    )
    device: str = "cuda"
    use_ema: bool = True

    # Robot configuration
    robot: RobotConfig | None = None

    # Action chunking & execution parameters
    chunk_size: int = 30
    n_action_steps: int = 15
    temporal_ensemble_coeff: float | None = None

    # Evaluation loop parameters
    task: str = DEFAULT_WIPES_TASK
    duration: float = 120.0
    fps: int = 30
    num_episodes: int = 5
    record_eval: bool = False
    push_to_hub: bool = False
    eval_dataset_repo_id: str = "temp/eval_turbovla_not_saved"
    eval_dataset_root: str | None = None
    display_data: bool = True

    def __post_init__(self) -> None:
        ckpt_p = Path(self.ckpt_path).expanduser().resolve()
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"TurboVLA checkpoint not found: {ckpt_p}")
        self.ckpt_path = str(ckpt_p)

        stats_p = Path(self.stats_path).expanduser().resolve()
        if not stats_p.is_file():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_p}")
        self.stats_path = str(stats_p)

        if self.robot is None:
            raise ValueError("Robot configuration must be provided (e.g. --robot.type=alicia_d_follower)")


@parser.wrap()
def eval_turbovla(cfg: TurboVLAEvalConfig) -> None:
    """Main entry point for TurboVLA evaluation on Alicia-D."""
    init_logging()
    logger.info("Starting TurboVLA evaluation on Alicia-D...")

    # Initialize policy first so model loading errors surface before moving hardware
    policy = TurboVLAAliciaPolicy(
        ckpt_path=cfg.ckpt_path,
        stats_path=cfg.stats_path,
        dinov3_path=cfg.dinov3_path,
        bert_path=cfg.bert_path,
        device=cfg.device,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,
        use_ema=cfg.use_ema,
        default_task=cfg.task,
    )
    preprocessor = _IdentityPolicyProcessor()
    postprocessor = _IdentityPolicyProcessor()

    # Initialize & connect robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    has_video_features = any("image" in key or "video" in key for key in dataset_features.keys())
    eval_root = Path(cfg.eval_dataset_root) if cfg.eval_dataset_root else None
    if not cfg.record_eval:
        stale_dir = (eval_root / cfg.eval_dataset_repo_id) if eval_root is not None else (HF_LEROBOT_HOME / cfg.eval_dataset_repo_id)
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
            logger.info(f"Removed stale temp dataset: {stale_dir}")

    eval_dataset = LeRobotDataset.create(
        repo_id=cfg.eval_dataset_repo_id,
        fps=int(round(cfg.fps)),
        features=dataset_features,
        robot_type=robot.name,
        use_videos=has_video_features,
        image_writer_threads=4 if cfg.record_eval else 0,
        root=eval_root / cfg.eval_dataset_repo_id if eval_root is not None else None,
    )

    listener, events = init_keyboard_listener()
    if cfg.display_data:
        init_rerun(session_name="alicia_d_turbovla_evaluate")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    logger.info(
        f"Starting evaluation loop: task={cfg.task!r}, chunk_size={cfg.chunk_size}, "
        f"n_action_steps={cfg.n_action_steps}, temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}"
    )
    recorded_episodes = 0

    try:
        while recorded_episodes < cfg.num_episodes and not events["stop_recording"]:
            log_say(f"Running TurboVLA inference, episode {recorded_episodes + 1} of {cfg.num_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=cfg.fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=eval_dataset,
                control_time_s=cfg.duration,
                single_task=cfg.task,
                display_data=cfg.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.fps,
                    control_time_s=cfg.duration,
                    single_task=cfg.task,
                    display_data=cfg.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if cfg.record_eval:
                    eval_dataset.clear_episode_buffer()
                continue

            if cfg.record_eval:
                eval_dataset.save_episode()
            recorded_episodes += 1

    finally:
        log_say("Stop evaluation")
        if robot.is_connected:
            robot.disconnect()
            logger.info("Robot disconnected")

        if not is_headless() and listener:
            listener.stop()

        if cfg.record_eval:
            eval_dataset.finalize()
            if cfg.push_to_hub:
                try:
                    eval_dataset.push_to_hub()
                except Exception as e:
                    logger.warning(f"Failed to push evaluation dataset to hub: {e}")

        logger.info("Evaluation finished")


def main() -> None:
    register_third_party_plugins()
    eval_turbovla()


if __name__ == "__main__":
    main()
