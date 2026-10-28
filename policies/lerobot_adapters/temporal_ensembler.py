#!/usr/bin/env python3
"""
ACTTemporalEnsembler: Exponentially weighted temporal smoothing over overlapping action chunks.
Used for real-robot deployment of ACT, SmolVLA, and other chunk-based imitation learning policies
to suppress joint velocity spikes and high-frequency jerk at chunk boundaries.
"""

from collections import deque
import torch
import numpy as np


class ACTTemporalEnsembler:
    """
    Maintains a FIFO queue of overlapping action chunks and aggregates predicted actions
    for the current time step using exponential decay weights:
        w_i = exp(-m * i)
    where:
        - i is the age of the chunk (0 for newest, 1 for previous step, etc.)
        - m is the temporal ensemble coefficient (e.g. 0.01)
    """

    def __init__(self, temporal_ensemble_coeff: float = 0.01, chunk_size: int = 30):
        self.coeff = temporal_ensemble_coeff
        self.chunk_size = chunk_size
        self.all_time_actions = deque(maxlen=chunk_size)
        # Precompute weights for up to chunk_size chunks
        weights = np.exp(-self.coeff * np.arange(self.chunk_size))
        self.weights = weights / np.sum(weights)

    def reset(self):
        """Clears accumulated chunk buffer upon episode reset."""
        self.all_time_actions.clear()

    def update(self, action_chunk: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Appends the latest predicted action chunk [chunk_size, action_dim]
        and returns the smoothed action [action_dim] for the current step.
        """
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.detach().cpu().numpy()

        if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk[0]  # [chunk_size, action_dim]

        self.all_time_actions.append(action_chunk)
        num_chunks = len(self.all_time_actions)

        # Collect actions predicted for the current timestep across chunks:
        # - newest chunk (age 0): prediction at index 0
        # - chunk from 1 step ago (age 1): prediction at index 1
        # - chunk from k steps ago (age k): prediction at index k
        actions_for_curr_step = []
        raw_weights = []
        for age, chunk in enumerate(reversed(self.all_time_actions)):
            actions_for_curr_step.append(chunk[age])
            raw_weights.append(np.exp(-self.coeff * age))

        actions_for_curr_step = np.stack(actions_for_curr_step, axis=0)  # [num_chunks, action_dim]
        norm_weights = np.array(raw_weights) / np.sum(raw_weights)       # [num_chunks]

        # Weighted combination: sum_i (w_i * a_i)
        smoothed_action = np.sum(actions_for_curr_step * norm_weights[:, None], axis=0)
        return smoothed_action
