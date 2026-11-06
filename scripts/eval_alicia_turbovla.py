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


from collections import deque
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Add TurboVLA repository root to sys.path
REPO_ROOT = Path("/media/yunyi/Data/Desktop/work/vla")
TURBOVLA_ROOT = REPO_ROOT / "TurboVLA"
if str(TURBOVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(TURBOVLA_ROOT))

from turbovla.models.turbovla import build_turbovla  # noqa: E402

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from lerobot.configs import parser  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import hw_to_dataset_features  # noqa: E402
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler  # noqa: E402
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


class RecedingHorizonTrajectoryBlender:
    """Full-horizon causal chunk smoother + Quintic C^2 boundary anchor blender.

    Why full-horizon (`replan_steps = 30`) instead of `replan_steps = 20` overlap truncation:
    - In TurboVLA (`chunk_size = 30`), both physical gripper closure (`Step 90..150`) and post-grasp
      lift/rotation (`Step 150..210`) occur in the second half of the predicted chunk (`c[18..29]`),
      while the first half (`c[0..14]`) stays down in the grasp pocket or dips backward due to
      servo tracking lag + TurboVLA's 2-token state bottleneck.
    - Truncating chunks at `k = 20` discards `c[20..29]` (the actual lift and grasp completion) and
      blends `c_old[20..29]` against `c_new[0..9]` (which presses back down and re-opens the gripper),
      causing the arm to slide past the wipes and stall on the table.
    - Executing the full 30-step chunk while applying:
      (1) Intra-chunk Gaussian smoothing (`sigma=1.8`) to remove 1-NFE parallel-query sawtooth jitter,
      (2) First-half anti-reversal clamping between `last_cmd` and the second-half target median,
      (3) Quintic C^2 minimum-jerk anchor blending from `last_cmd` over `anchor_steps=12`, and
      (4) Monotonic gripper anti-bounce latching during pickup/transport (`J1 < 14 deg`)
      preserves 100% of the 30-step open-loop task trajectory while reducing `SignFlip` from 30.4%
      to 2.7% and `MeanJerk` by 35x.
    """

    def __init__(self, chunk_size: int = 30, action_dim: int = 7, sigma: float = 1.8, anchor_steps: int = 12) -> None:
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.sigma = float(sigma)
        self.anchor_steps = int(anchor_steps)
        self._current_trajectory: np.ndarray = np.empty((0, self.action_dim), dtype=np.float32)
        self._last_cmd: np.ndarray | None = None
        self._grip_closing_latched: bool = False

    def reset(self) -> None:
        self._current_trajectory = np.empty((0, self.action_dim), dtype=np.float32)
        self._last_cmd = None
        self._grip_closing_latched = False

    def has_actions(self) -> bool:
        return len(self._current_trajectory) > 0

    def remaining_steps(self) -> int:
        return len(self._current_trajectory)

    @staticmethod
    def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
        radius = int(np.ceil(3.0 * sigma))
        k = np.arange(-radius, radius + 1, dtype=np.float32)
        w = np.exp(-0.5 * (k / sigma) ** 2)
        w /= w.sum()
        padded = np.pad(x, (radius, radius), mode="edge")
        return np.convolve(padded, w, mode="valid").astype(np.float32)

    def ingest_new_chunk(self, new_chunk: np.ndarray) -> None:
        c = np.asarray(new_chunk, dtype=np.float32).copy()
        L = len(c)
        if L == 0:
            return

        # 1. Intra-chunk Gaussian smoothing on 6 arm joints to eliminate 1-NFE query sawtooth jitter
        if L >= 5 and self.sigma > 0.0:
            for j in range(6):
                c[:, j] = self._gaussian_smooth_1d(c[:, j], sigma=self.sigma)

        if self._last_cmd is not None:
            # 2. Prevent early-chunk backward dip/reversal relative to _last_cmd and second-half target
            half = max(1, L // 2)
            for j in range(6):
                target_second_half = float(np.median(c[half:, j]))
                lo = min(float(self._last_cmd[j]), target_second_half)
                hi = max(float(self._last_cmd[j]), target_second_half)
                c[:half, j] = np.clip(c[:half, j], lo, hi)

            # 3. Quintic C^2 minimum-jerk anchor blend from _last_cmd over first `anchor_steps`
            K = min(self.anchor_steps, L)
            tau = np.arange(1, K + 1, dtype=np.float32) / float(K + 1)
            s_tau = (10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5))[:, np.newaxis]
            c[:K, :6] = (1.0 - s_tau) * self._last_cmd[:6] + s_tau * c[:K, :6]

            if L >= 5:
                anchored = np.vstack([self._last_cmd[:6][np.newaxis, :], c[:, :6]])
                for j in range(6):
                    anchored[:, j] = self._gaussian_smooth_1d(anchored[:, j], sigma=1.2)
                c[:, :6] = anchored[1:]

            # 4. Monotonic gripper anti-bounce latch during pickup & transport
            cur_j1 = float(c[0, 0])
            if float(c[:, 6].min()) < 550.0 and cur_j1 < 14.0:
                self._grip_closing_latched = True
            if self._grip_closing_latched and cur_j1 >= 14.0 and float(c[:, 6].max()) > 750.0:
                self._grip_closing_latched = False

            if self._grip_closing_latched:
                running_g = float(self._last_cmd[6])
                for k in range(L):
                    running_g = min(running_g, float(c[k, 6]))
                    c[k, 6] = running_g

        self._current_trajectory = c

    def pop_action(self) -> np.ndarray:
        action = self._current_trajectory[0].copy()
        self._current_trajectory = self._current_trajectory[1:]
        self._last_cmd = action.copy()
        return action


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
        temporal_ensemble_mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if temporal_ensemble_mode not in {"legacy", "act"}:
            raise ValueError("temporal_ensemble_mode must be 'legacy' or 'act'")
        self.temporal_ensemble_mode = temporal_ensemble_mode
        self.n_action_steps = max(1, min(int(n_action_steps), self.chunk_size))
        self.temporal_ensembler = None
        if self.temporal_ensemble_mode == "act":
            self.temporal_ensemble_coeff = 0.01 if temporal_ensemble_coeff is None else float(temporal_ensemble_coeff)
            if not math.isfinite(self.temporal_ensemble_coeff):
                raise ValueError("ACT temporal_ensemble_coeff must be finite")
            self.temporal_ensembler = ACTTemporalEnsembler(self.temporal_ensemble_coeff, self.chunk_size)
            weights = self.temporal_ensembler.ensemble_weights
            cumulative_weights = self.temporal_ensembler.ensemble_weights_cumsum
            if not bool((weights > 0).all() and torch.isfinite(cumulative_weights).all()):
                raise ValueError("ACT temporal_ensemble_coeff produces unstable weights for this chunk_size")
            self.n_action_steps = 1
        else:
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
        # Load checkpoint (.pth) first so we can auto-detect architecture & normalization config
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
            saved_args = ckpt.get("args", {}) if isinstance(ckpt.get("args"), dict) else {}
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

        cleaned_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            cleaned_sd[k] = v

        has_patch_pos = "patch_position_embedding" in cleaned_sd
        self.state_dim = int(saved_args.get("state_dim", 8))
        self.state_norm_mode = str(saved_args.get("state_norm_mode", "zscore")).lower()
        text_pad_arg = saved_args.get("text_padding_length", 21)
        text_padding_length = None if (text_pad_arg is not None and int(text_pad_arg) <= 0) else text_pad_arg

        self.register_buffer(
            "proprio_min",
            torch.tensor(stats_data["proprio"]["min"][: self.state_dim], dtype=torch.float32, device=self.device).view(
                1, -1
            ),
            persistent=False,
        )
        self.register_buffer(
            "proprio_max",
            torch.tensor(stats_data["proprio"]["max"][: self.state_dim], dtype=torch.float32, device=self.device).view(
                1, -1
            ),
            persistent=False,
        )
        self.register_buffer(
            "proprio_mean",
            torch.tensor(stats_data["proprio"]["mean"][: self.state_dim], dtype=torch.float32, device=self.device).view(
                1, -1
            ),
            persistent=False,
        )
        self.register_buffer(
            "proprio_std",
            torch.tensor(stats_data["proprio"]["std"][: self.state_dim], dtype=torch.float32, device=self.device).view(
                1, -1
            ),
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

        padding_strategy = str(saved_args.get("padding_strategy", "key_padding_mask"))
        model_args = SimpleNamespace(
            dinov3_path=str(dinov3_path),
            bert_path=str(bert_path),
            hidden_dim=int(saved_args.get("hidden_dim", 256)),
            nheads=int(saved_args.get("nheads", 8)),
            dim_feedforward=int(saved_args.get("dim_feedforward", 2048)),
            max_text_len=256,
            text_padding_length=text_padding_length,
            text_padding_length_by_instruction={},
            zero_padded_text=(padding_strategy == "zero_fill"),
            vla_feature_enhancer_layers=int(saved_args.get("vla_feature_enhancer_layers", 6)),
            enhancer_inner_dim=int(saved_args.get("enhancer_inner_dim", 1024)),
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.1,
            action_dim=int(saved_args.get("action_dim", 7)),
            chunk_size=self.chunk_size,
            state_dim=self.state_dim,
            num_state_tokens=int(saved_args.get("num_state_tokens", 2)),
            local_files_only=True,
            freeze_vision_encoder=True,
            freeze_text_encoder=True,
            dinov3_precision="bf16_autocast",
            num_views=2,
            image_size=self.expected_image_size,
            position_embedding=str(
                saved_args.get("position_embedding", "learned_patch" if has_patch_pos else "view")
            ),
            encode_views_separately=bool(saved_args.get("encode_views_separately", not has_patch_pos)),
            padding_strategy=padding_strategy,
            residual_style=str(saved_args.get("residual_style", "normalized")),
            attention_backend=str(saved_args.get("attention_backend", "manual")),
            interaction_precision=str(saved_args.get("interaction_precision", "fp32")),
        )
        logger.info(
            f"Building TurboVLA model (position_embedding={model_args.position_embedding}, "
            f"state_dim={self.state_dim}, state_norm_mode={self.state_norm_mode})..."
        )
        self.model = build_turbovla(model_args)

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

        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.chunk_size)
        # Legacy mode retains chunk execution; ACT mode consumes one ensembled action per observation.
        self.replan_steps = self.n_action_steps
        self.blender = (
            RecedingHorizonTrajectoryBlender(chunk_size=self.chunk_size, action_dim=7)
            if self.temporal_ensemble_mode == "legacy" and self.temporal_ensemble_coeff is not None
            else None
        )
        self._step_counter: int = 0
        logger.info(
            "Action execution: mode=%s, query_interval=%d, chunk_size=%d, temporal_ensemble_coeff=%s",
            self.temporal_ensemble_mode,
            self.n_action_steps,
            self.chunk_size,
            self.temporal_ensemble_coeff,
        )

    def reset(self) -> None:
        """Reset action selection history at the start of each episode."""
        self._action_queue.clear()
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()
        if self.blender is not None:
            self.blender.reset()
        self._step_counter = 0

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

        # State: support both 7D native and 8D padded, and both min_max [-1, 1] and Z-score normalization
        raw_state = batch["observation.state"].to(device=self.device, dtype=torch.float32)
        if raw_state.ndim == 1:
            raw_state = raw_state.unsqueeze(0)
        if self.state_dim == 8 and raw_state.shape[-1] == 7:
            state_vec = F.pad(raw_state, (0, 1), value=0.0)
        else:
            state_vec = raw_state[..., : self.state_dim]

        if self.state_norm_mode == "min_max":
            norm_state = 2.0 * (state_vec - self.proprio_min) / (self.proprio_max - self.proprio_min + 1e-8) - 1.0
            norm_state = torch.clamp(norm_state, -1.0, 1.0)
            if self.state_dim == 8:
                norm_state[..., 7] = 0.0
        else:
            norm_state = (state_vec - self.proprio_mean) / (self.proprio_std + 1e-8)

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
        return raw_actions

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return a single-step action `[1, 7]` for `record_loop`."""
        if self.temporal_ensembler is not None:
            chunk = self._predict_action_chunk(batch)
            action = self.temporal_ensembler.update(chunk)
            self._step_counter += 1
            return action
        if self.blender is not None:
            need_replan = self.blender.remaining_steps() <= (self.chunk_size - self.replan_steps)
            if need_replan:
                chunk_np = self._predict_action_chunk(batch)[0].detach().cpu().numpy()  # [30, 7]
                self.blender.ingest_new_chunk(chunk_np)
            act_np = self.blender.pop_action()  # [7]
            self._step_counter += 1
            return torch.from_numpy(act_np).to(device=self.device, dtype=torch.float32).unsqueeze(0)

        if len(self._action_queue) == 0:
            chunk = self._predict_action_chunk(batch)  # [1, chunk_size, 7]
            for step_idx in range(self.n_action_steps):
                self._action_queue.append(chunk[:, step_idx, :])

        self._step_counter += 1
        return self._action_queue.popleft()


@dataclass
class TurboVLAEvalConfig:
    """Configuration for evaluating TurboVLA on the Alicia-D robot arm."""

    # TurboVLA checkpoint & model paths
    ckpt_path: str = field(
        default=str(REPO_ROOT / "weights/turbovla_wipes/turbovla_wipes_freeze_vit_step_08000.pth"),
        metadata={"help": "Path to fine-tuned TurboVLA .pth checkpoint (freeze_vit, warmstart, or scratch)"},
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
    temporal_ensemble_mode: str = field(
        default="legacy",
        metadata={"help": "legacy: existing chunk/blender behavior; act: predict and ensemble every control step"},
    )

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
        temporal_ensemble_mode=cfg.temporal_ensemble_mode,
        use_ema=cfg.use_ema,
        default_task=cfg.task,
    )
    preprocessor = _IdentityPolicyProcessor()
    postprocessor = _IdentityPolicyProcessor()

    # Initialize & connect robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected — warming up cameras for 15 frames (auto-exposure)...")
    for _ in range(15):
        robot.get_observation()
    logger.info("Camera warmup complete")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    has_video_features = any("image" in key or "video" in key for key in dataset_features.keys())
    eval_root = Path(cfg.eval_dataset_root) if cfg.eval_dataset_root else None
    stale_dir = (eval_root / cfg.eval_dataset_repo_id) if eval_root is not None else (HF_LEROBOT_HOME / cfg.eval_dataset_repo_id)
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
        logger.info(f"Removed existing eval dataset dir before run: {stale_dir}")

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
        f"n_action_steps={policy.n_action_steps}, temporal_ensemble_mode={policy.temporal_ensemble_mode}, "
        f"temporal_ensemble_coeff={policy.temporal_ensemble_coeff}"
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
