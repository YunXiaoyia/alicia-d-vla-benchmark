#!/usr/bin/env python3
"""
CameraMapper: Observation feature namespace adapter.
Maps physical camera observation keys (e.g., 'top', 'wrist') into model-specific
expected keys across different VLA model architectures (Pi0.5, SmolVLA, ACT, TurboVLA).
"""

from typing import Any, Dict


POLICY_KEY_MAPPINGS = {
    "pi05": {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    },
    "pi0": {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    },
    "smolvla": {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    },
    "act": {
        # Native LeRobot naming
        "observation.images.top": "observation.images.top",
        "observation.images.wrist": "observation.images.wrist",
    },
    "turbovla": {
        "observation.images.top": "top",
        "observation.images.wrist": "wrist",
    },
}


class CameraMapper:
    """Performs zero-copy observation dictionary remapping for heterogeneous policies."""

    def __init__(self, policy_type: str, custom_map: Dict[str, str] | None = None):
        self.policy_type = policy_type.lower()
        self.mapping = custom_map or POLICY_KEY_MAPPINGS.get(self.policy_type, {})

    def remap_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a new dictionary containing the remapped observation keys.
        Preserves unmapped keys unmodified.
        """
        if not self.mapping:
            return observation

        remapped = {}
        for k, v in observation.items():
            target_k = self.mapping.get(k, k)
            remapped[target_k] = v
        return remapped
