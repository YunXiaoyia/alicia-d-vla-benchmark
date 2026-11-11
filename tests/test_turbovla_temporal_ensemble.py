"""Offline checks for TurboVLA action selection."""

from collections import deque
import math
import unittest

import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from scripts.eval_alicia_turbovla import RecedingHorizonTrajectoryBlender, TurboVLAAliciaPolicy


def make_policy(chunks, mode="act", coefficient=0.01):
    policy = TurboVLAAliciaPolicy.__new__(TurboVLAAliciaPolicy)
    nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.chunk_size = 3
    policy.n_action_steps = 3
    policy.replan_steps = 3
    policy.temporal_ensemble_mode = mode
    policy.temporal_ensembler = ACTTemporalEnsembler(coefficient, policy.chunk_size) if mode == "act" else None
    policy.blender = None
    policy._action_queue = deque(maxlen=policy.chunk_size)
    policy._step_counter = 0
    calls = []

    def predict(batch):
        calls.append(batch)
        return torch.tensor(chunks[len(calls) - 1], dtype=torch.float32).view(1, 3, 1).expand(1, 3, 7)

    policy._predict_action_chunk = predict
    return policy, calls


class TurboVLATemporalEnsembleTest(unittest.TestCase):
    def test_act_mode_aligns_overlapping_chunks_and_predicts_every_step(self):
        policy, calls = make_policy([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]])
        outputs = [policy.select_action({}).squeeze()[0].item() for _ in range(4)]
        weights = [math.exp(-0.01 * i) for i in range(3)]
        expected = [
            10,
            (11 * weights[0] + 20 * weights[1]) / sum(weights[:2]),
            (12 * weights[0] + 21 * weights[1] + 30 * weights[2]) / sum(weights),
            (22 * weights[0] + 31 * weights[1] + 40 * weights[2]) / sum(weights),
        ]
        self.assertEqual(len(calls), 4)
        for actual, target in zip(outputs, expected):
            self.assertAlmostEqual(actual, target, places=5)

    def test_act_mode_accepts_uniform_weights_and_resets_between_episodes(self):
        policy, calls = make_policy([[10, 11, 12], [20, 21, 22], [40, 41, 42]], coefficient=0.0)
        self.assertEqual(policy.select_action({}).squeeze()[0].item(), 10)
        self.assertAlmostEqual(policy.select_action({}).squeeze()[0].item(), 15.5)
        policy.reset()
        self.assertEqual(policy.select_action({}).squeeze()[0].item(), 40)
        self.assertEqual(len(calls), 3)

    def test_legacy_queue_keeps_existing_replan_interval(self):
        policy, calls = make_policy([[10, 11, 12], [20, 21, 22]], mode="legacy")
        outputs = [policy.select_action({}).squeeze()[0].item() for _ in range(4)]
        self.assertEqual(outputs, [10, 11, 12, 20])
        self.assertEqual(len(calls), 2)

    def test_legacy_blender_keeps_existing_replan_interval(self):
        policy, calls = make_policy([[10, 11, 12], [20, 21, 22]], mode="legacy")
        policy.blender = RecedingHorizonTrajectoryBlender(chunk_size=3, action_dim=7)
        outputs = [policy.select_action({}).squeeze()[0].item() for _ in range(4)]
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
