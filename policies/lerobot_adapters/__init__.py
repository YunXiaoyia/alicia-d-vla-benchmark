"""
LeRobot Policy Adapters & Training Recipes for Alicia-D VLA Benchmark.
Includes training scripts, LoRA merge utilities, camera observation namespace adapters,
and real-robot temporal smoothing modules (Temporal Ensemble, RTC).
"""

from .temporal_ensembler import ACTTemporalEnsembler
from .camera_mapper import CameraMapper, POLICY_KEY_MAPPINGS
from .rtc_processor import RTCProcessor

__all__ = [
    "ACTTemporalEnsembler",
    "CameraMapper",
    "POLICY_KEY_MAPPINGS",
    "RTCProcessor",
]
