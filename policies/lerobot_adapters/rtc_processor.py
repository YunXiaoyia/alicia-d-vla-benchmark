#!/usr/bin/env python3
"""
RTCProcessor: Real-Time Chunking (RTC) Prefix Guidance Processor for Flow Matching VLAs.
Provides smooth action chunk blending for diffusion/flow policies (such as Pi0.5) by constraining
new chunk trajectories to match remaining actions from the previous inference cycle.
"""

import torch
import numpy as np


class RTCProcessor:
    """
    Tracks action execution within a chunk and computes the left-over prefix
    for the subsequent chunk's ODE sampling.
    """

    def __init__(self, chunk_size: int = 30, execution_horizon: int = 15, max_guidance_weight: float = 10.0):
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.max_guidance_weight = max_guidance_weight
        self.prev_chunk_left_over = None

    def reset(self):
        """Reset state at episode start."""
        self.prev_chunk_left_over = None

    def prepare_guidance_prefix(self, current_chunk: torch.Tensor | np.ndarray) -> torch.Tensor | None:
        """
        Extracts the unexecuted slice of current_chunk [chunk_size, action_dim]
        to be passed as guidance into the next policy call.
        """
        if isinstance(current_chunk, np.ndarray):
            current_chunk = torch.from_numpy(current_chunk)

        # Remaining steps beyond execution_horizon
        if self.execution_horizon < self.chunk_size:
            self.prev_chunk_left_over = current_chunk[self.execution_horizon:].clone()
        else:
            self.prev_chunk_left_over = None

        return self.prev_chunk_left_over

    def get_guidance_kwargs(self) -> dict:
        """Returns kwargs to inject into the policy forward pass."""
        if self.prev_chunk_left_over is None:
            return {}
        return {
            "prev_chunk_left_over": self.prev_chunk_left_over,
            "guidance_weight": self.max_guidance_weight,
        }
