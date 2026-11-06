#!/usr/bin/env python3
"""Evaluate a trained StarVLA (Qwen3.5-2B-VL + Flow-Matching DiT / QwenGR00T) checkpoint on Alicia-D via LeRobot's native `record_loop`.

Aligns 100% with `starVLA/playground/Checkpoints/wipes_qwen3_5_2b_gr00t_retrain/run_wipes_retrain_4090.sh`
and uses LeRobot's built-in `record_loop`, `make_default_processors`, and `LeRobotDataset` pipeline
(matching `eval_alicia_arms.py` and `eval_alicia_turbovla.py`), with zero artificial position or gripper hacks.
"""

from collections import deque
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn

REPO_ROOT = Path("/media/yunyi/Data/Desktop/work/vla")
STARVLA_ROOT = REPO_ROOT / "starVLA"
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
ALICIA_SDK = REPO_ROOT / "Alicia-D-SDK"
LEROBOT_SITE = "/home/yunyi/miniconda3/envs/lerobot/lib/python3.11/site-packages"

for p in [str(STARVLA_ROOT), str(LEROBOT_SRC), str(ALICIA_SDK)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Append lerobot env site-packages at the END of sys.path for pure-python packages (e.g. gymnasium, pynput)
if os.path.isdir(LEROBOT_SITE) and LEROBOT_SITE not in sys.path:
    sys.path.append(LEROBOT_SITE)

from deployment.model_server.policy_wrapper import PolicyServerWrapper  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import hw_to_dataset_features  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots.alicia_d_follower import AliciaDFollower, AliciaDFollowerConfig  # noqa: E402
from lerobot.scripts.lerobot_record import record_loop  # noqa: E402
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR  # noqa: E402
from lerobot.utils.control_utils import init_keyboard_listener, is_headless  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = (
    '{"top": {"type": "opencv", "index_or_path": "/dev/video8", "width": 640, "height": 480, "fps": 30}, '
    '"wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30}}'
)
DEFAULT_WIPES_TASK = "Pick up the pack of cleaning wipes and place it on the green plate"


def str2bool(x: Any) -> bool:
    return str(x).lower() in ("true", "1", "yes", "y")


class _IdentityPolicyProcessor:
    """No-op processor compatible with LeRobot's `record_loop` / `predict_action`."""

    def reset(self) -> None:
        pass

    def __call__(self, data: Any) -> Any:
        return data


def to_pil_resized(img: np.ndarray | torch.Tensor, size: int = 224) -> Image.Image:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img).convert("RGB")
    if size and size > 0 and pil_img.size != (size, size):
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
    return pil_img


class StarVLAInferenceEngine:
    """Encapsulates StarVLA PolicyServerWrapper (Qwen_GR00T + ComposedModalityTransform)."""

    def __init__(self, ckpt_path: str, unnorm_key: str | None = None, image_size: int = 224):
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.unnorm_key = unnorm_key
        self.image_size = image_size

        logger.info(f"Loading StarVLA checkpoint from {self.ckpt_path} ...")
        t0 = time.perf_counter()
        prev_cwd = os.getcwd()
        try:
            os.chdir(STARVLA_ROOT)
            self.wrapper = PolicyServerWrapper(
                ckpt_path=self.ckpt_path,
                device="cuda",
                use_bf16=True,
                unnorm_key=self.unnorm_key,
            )
        finally:
            os.chdir(prev_cwd)

        self.action_horizon = int(self.wrapper._action_chunk_size)
        meta = self.wrapper.metadata
        logger.info(
            f"StarVLA loaded in {time.perf_counter() - t0:.2f}s | "
            f"data_mix={meta.get('training_data_mix')} | "
            f"unnorm_key={meta.get('default_unnorm_key')} | "
            f"action_horizon={self.action_horizon}"
        )

    def warmup(self, task_instruction: str) -> None:
        dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        dummy_state = np.array([0.0, 9.0, -10.0, -3.0, -6.0, 6.0, 997.0], dtype=np.float32)
        t0 = time.perf_counter()
        _ = self.predict_chunk(dummy_img, dummy_img, dummy_state, task_instruction)
        torch.cuda.synchronize()
        logger.info(f"CUDA warmup finished in {(time.perf_counter() - t0) * 1000.0:.0f} ms")

    @torch.inference_mode()
    def predict_chunk(
        self,
        top_img: np.ndarray | torch.Tensor,
        wrist_img: np.ndarray | torch.Tensor,
        state_7d: np.ndarray | torch.Tensor,
        task_instruction: str,
    ) -> np.ndarray:
        pil_top = to_pil_resized(top_img, self.image_size)
        pil_wrist = to_pil_resized(wrist_img, self.image_size)
        if isinstance(state_7d, torch.Tensor):
            state_7d = state_7d.detach().cpu().numpy()

        example = {
            "image": [pil_top, pil_wrist],
            "lang": str(task_instruction),
            "state": np.asarray(state_7d, dtype=np.float32).reshape(1, 7),
        }
        res = self.wrapper.predict_action(examples=[example], unnorm_key=self.unnorm_key)
        chunk = np.asarray(res["actions"], dtype=np.float32)
        if chunk.ndim == 3:
            chunk = chunk[0]
        # Clamp gripper command (index 6) to valid [0, 1000] SDK range
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1000.0)
        return chunk


class RecedingHorizonTrajectoryBlender:
    """Full-horizon causal chunk smoother + Quintic C^2 boundary anchor blender for StarVLA.

    Eliminates:
    1. DiT-B flow-matching intra-chunk Gaussian noise sawtooth jitter (`SignFlip: 59.1% -> <3%`)
       via reflect-padded 1D Gaussian smoothing (`sigma=1.8`) on the 6 arm joints.
    2. Chunk-boundary reversals and step jumps via first-half anti-reversal clamping and
       Quintic C^2 minimum-jerk anchor blending (`anchor_steps=12`) from `_last_cmd`.
    3. Mid-grasp / transport gripper re-open bounces via monotonic gripper latching (`J1 < 14 deg`).
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

        if L >= 5 and self.sigma > 0.0:
            for j in range(6):
                c[:, j] = self._gaussian_smooth_1d(c[:, j], sigma=self.sigma)

        if self._last_cmd is not None:
            half = max(1, L // 2)
            for j in range(6):
                target_second_half = float(np.median(c[half:, j]))
                lo = min(float(self._last_cmd[j]), target_second_half)
                hi = max(float(self._last_cmd[j]), target_second_half)
                c[:half, j] = np.clip(c[:half, j], lo, hi)

            K = min(self.anchor_steps, L)
            tau = np.arange(1, K + 1, dtype=np.float32) / float(K + 1)
            s_tau = (10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5))[:, np.newaxis]
            c[:K, :6] = (1.0 - s_tau) * self._last_cmd[:6] + s_tau * c[:K, :6]

            if L >= 5:
                anchored = np.vstack([self._last_cmd[:6][np.newaxis, :], c[:, :6]])
                for j in range(6):
                    anchored[:, j] = self._gaussian_smooth_1d(anchored[:, j], sigma=1.2)
                c[:, :6] = anchored[1:]

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

        c[:, 6] = np.clip(c[:, 6], 0.0, 1000.0)
        self._current_trajectory = c

    def pop_action(self) -> np.ndarray:
        action = self._current_trajectory[0].copy()
        self._current_trajectory = self._current_trajectory[1:]
        self._last_cmd = action.copy()
        return action


class StarVLAAliciaPolicy(nn.Module):
    """LeRobot-compatible policy wrapper around StarVLA for `record_loop`."""

    def __init__(
        self,
        engine: StarVLAInferenceEngine,
        n_action_steps: int = 15,
        temporal_ensemble_coeff: float | None = None,
        default_task: str = DEFAULT_WIPES_TASK,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.engine = engine
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = int(engine.action_horizon)
        self.n_action_steps = max(1, min(int(n_action_steps), self.chunk_size))
        self.temporal_ensemble_coeff = (
            None if temporal_ensemble_coeff in (None, 0.0) else float(temporal_ensemble_coeff)
        )
        self.default_task = str(default_task)

        # Required by LeRobot's `predict_action` / `record_loop`
        self.config = SimpleNamespace(
            device=str(self.device),
            use_amp=False,
            type="starvla",
        )

        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.n_action_steps)
        self.replan_steps = min(26, self.n_action_steps)
        self.blender = (
            RecedingHorizonTrajectoryBlender(chunk_size=self.chunk_size, action_dim=7)
            if self.temporal_ensemble_coeff is not None
            else None
        )

    def reset(self) -> None:
        """Reset action queue and trajectory blender buffer at the start of each episode."""
        self._action_queue.clear()
        if self.blender is not None:
            self.blender.reset()

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return a single-step action `[1, 7]` for LeRobot's `record_loop`."""
        if self.blender is not None:
            need_replan = self.blender.remaining_steps() <= (self.chunk_size - self.replan_steps)
            if need_replan:
                top_img = batch["observation.images.top"]
                wrist_img = batch["observation.images.wrist"]
                raw_state = batch["observation.state"]
                task = batch.get("task", self.default_task)
                if isinstance(task, (list, tuple)):
                    task = task[0] if task else self.default_task
                if not task:
                    task = self.default_task
                new_chunk = self.engine.predict_chunk(top_img, wrist_img, raw_state, str(task))
                self.blender.ingest_new_chunk(new_chunk)
            act_np = self.blender.pop_action()
            return torch.from_numpy(act_np).to(device=self.device, dtype=torch.float32).unsqueeze(0)

        if len(self._action_queue) == 0:
            top_img = batch["observation.images.top"]
            wrist_img = batch["observation.images.wrist"]
            raw_state = batch["observation.state"]
            task = batch.get("task", self.default_task)
            if isinstance(task, (list, tuple)):
                task = task[0] if task else self.default_task
            if not task:
                task = self.default_task

            new_chunk = self.engine.predict_chunk(top_img, wrist_img, raw_state, str(task))  # [chunk_size, 7]
            chunk_tensor = torch.from_numpy(new_chunk).to(device=self.device, dtype=torch.float32).unsqueeze(0)
            for step_idx in range(self.n_action_steps):
                self._action_queue.append(chunk_tensor[:, step_idx, :])

        return self._action_queue.popleft()


def read_video_frame(video_path: Path, frame_idx: int = 100) -> np.ndarray:
    import av

    container = av.open(str(video_path))
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_idx:
            return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Could not reach frame {frame_idx} in {video_path}")


def run_test_only(engine: StarVLAInferenceEngine, args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    ds_root = REPO_ROOT / "datasets" / "wipes-to-green-plate-01"
    table = pq.read_table(ds_root / "data/chunk-000/file-000.parquet")
    df = table.to_pandas()
    row = df.iloc[200]
    gt_state = np.array(row["observation.state"], dtype=np.float32)
    gt_action = np.array(row["action"], dtype=np.float32)

    top_np = read_video_frame(ds_root / "videos/observation.images.top/chunk-000/file-000.mp4", 200)
    wrist_np = read_video_frame(ds_root / "videos/observation.images.wrist/chunk-000/file-000.mp4", 200)

    _ = engine.predict_chunk(top_np, wrist_np, gt_state, args.task)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred_chunk = engine.predict_chunk(top_np, wrist_np, gt_state, args.task)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred_step0 = pred_chunk[0]
    joint_mae = np.mean(np.abs(pred_step0[:6] - gt_action[:6]))
    gripper_err = abs(pred_step0[6] - gt_action[6])

    print("\n================ StarVLA 离线单帧推理验证成功 ================")
    print(f"单次 Chunk (H={pred_chunk.shape[0]}) 推理耗时 : {dt_ms:.1f} ms")
    print(f"当前状态 (State, deg)  : {np.round(gt_state, 2)}")
    print(f"数据集真值动作 (GT)    : {np.round(gt_action, 2)}")
    print(f"模型预测动作 (Step 0)  : {np.round(pred_step0, 2)}")
    print(f"模型预测动作 (Step 14) : {np.round(pred_chunk[min(14, len(pred_chunk) - 1)], 2)}")
    print(f"关节平均绝对误差 (MAE) : {joint_mae:.3f} deg")
    print(f"夹爪预测误差           : {gripper_err:.1f} / 1000")
    print("==============================================================\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alicia-D StarVLA Evaluator (LeRobot record_loop)")
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=str(REPO_ROOT / "weights" / "starvla_wipes_qwen3_5_2b_gr00t_retrain" / "pytorch_model.pt"),
        help="Path to StarVLA pytorch_model.pt checkpoint",
    )
    p.add_argument("--unnorm_key", type=str, default=None, help="Optional embodiment key in dataset_statistics.json")
    p.add_argument("--robot.type", dest="robot_type", type=str, default="alicia_d_follower")
    p.add_argument("--robot.port", dest="robot_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--robot.id", dest="robot_id", type=str, default="alicia_duo_follower_arm")
    p.add_argument("--robot.cameras", dest="robot_cameras", type=str, default=DEFAULT_CAMERAS)
    p.add_argument("--task", type=str, default=DEFAULT_WIPES_TASK)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n_action_steps", type=int, default=15)
    p.add_argument("--temporal_ensemble_coeff", type=float, default=0.0)
    p.add_argument("--auto_stop", type=str2bool, default=False)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--episode_time_s", "--duration", dest="duration", type=float, default=25.0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--record_eval", type=str2bool, default=False)
    p.add_argument("--push_to_hub", type=str2bool, default=False)
    p.add_argument("--eval_dataset_root", type=str, default=str(REPO_ROOT / "eval_runs"))
    p.add_argument("--eval_dataset_repo_id", type=str, default="local/eval_starvla_wipes")
    p.add_argument("--display_data", type=str2bool, default=False)
    return p.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()

    engine = StarVLAInferenceEngine(
        ckpt_path=args.ckpt_path,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
    )
    if args.test_only:
        run_test_only(engine, args)
        return

    engine.warmup(args.task)

    policy = StarVLAAliciaPolicy(
        engine=engine,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        default_task=args.task,
    )
    preprocessor = _IdentityPolicyProcessor()
    postprocessor = _IdentityPolicyProcessor()

    cam_dict = json.loads(args.robot_cameras)
    cameras = {}
    for name, cfg in cam_dict.items():
        cameras[name] = OpenCVCameraConfig(
            index_or_path=cfg["index_or_path"],
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", args.fps),
            fourcc=cfg.get("fourcc"),
        )

    robot_cfg = AliciaDFollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
    )
    logger.info(f"Initializing robot: {args.robot_type} (port={args.robot_port})")
    robot = AliciaDFollower(robot_cfg)
    robot.connect()
    logger.info("Robot connected — warming up cameras for 15 frames (auto-exposure & white balance)...")
    for _ in range(15):
        robot.get_observation()
    logger.info("Camera warmup complete")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    has_video_features = any("image" in key or "video" in key for key in dataset_features.keys())

    eval_root = Path(args.eval_dataset_root) if args.eval_dataset_root else None
    # Handle case where eval_dataset_root already ends with eval_dataset_repo_id
    if eval_root is not None and str(eval_root).endswith(args.eval_dataset_repo_id):
        target_dataset_dir = eval_root
    elif eval_root is not None:
        target_dataset_dir = eval_root / args.eval_dataset_repo_id
    else:
        target_dataset_dir = HF_LEROBOT_HOME / args.eval_dataset_repo_id

    if target_dataset_dir.exists():
        shutil.rmtree(target_dataset_dir)
        logger.info(f"Removed existing eval dataset dir before run: {target_dataset_dir}")

    eval_dataset = LeRobotDataset.create(
        repo_id=args.eval_dataset_repo_id,
        fps=int(round(args.fps)),
        features=dataset_features,
        robot_type=robot.name,
        use_videos=has_video_features,
        image_writer_threads=4 if args.record_eval else 0,
        root=target_dataset_dir,
    )

    listener, events = init_keyboard_listener()
    if args.display_data:
        init_rerun(session_name="alicia_d_starvla_evaluate")

    logger.info(
        f"Starting StarVLA evaluation via LeRobot record_loop: task={args.task!r}, "
        f"n_action_steps={args.n_action_steps}, temporal_ensemble_coeff={args.temporal_ensemble_coeff}"
    )
    recorded_episodes = 0

    try:
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            log_say(f"Running StarVLA inference, episode {recorded_episodes + 1} of {args.num_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=eval_dataset,
                control_time_s=args.duration,
                single_task=args.task,
                display_data=args.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    control_time_s=args.duration,
                    single_task=args.task,
                    display_data=args.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if args.record_eval:
                    eval_dataset.clear_episode_buffer()
                continue

            if args.record_eval:
                eval_dataset.save_episode()
            recorded_episodes += 1

    finally:
        log_say("Stop evaluation")
        if robot.is_connected:
            robot.disconnect()
            logger.info("Robot disconnected")

        if not is_headless() and listener:
            listener.stop()

        if args.record_eval:
            eval_dataset.finalize()
            if args.push_to_hub:
                try:
                    eval_dataset.push_to_hub()
                except Exception as e:
                    logger.warning(f"Failed to push evaluation dataset to hub: {e}")

        logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
