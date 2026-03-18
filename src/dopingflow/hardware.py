from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


DeviceMode = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class HardwareConfig:
    device: DeviceMode = "auto"
    gpu_id: int = 0
    allow_gpu_batching: bool = True
    relax_batch_size: int = 8
    bandgap_batch_size: int = 32


def resolve_torch_device(mode: str = "auto", gpu_id: int = 0) -> str:
    import torch

    if mode == "cpu":
        return "cpu"

    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return f"cuda:{gpu_id}"

    # auto
    return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"


def configure_tensorflow(mode: str = "auto", gpu_id: int = 0) -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if mode == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "cpu"

    if mode == "cuda":
        if not gpus:
            raise RuntimeError("CUDA was requested, but TensorFlow sees no GPU.")
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        return f"cuda:{gpu_id}"

    # auto
    if gpus:
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        return f"cuda:{gpu_id}"

    return "cpu"


def parse_hardware_config(raw_cfg: dict) -> HardwareConfig:
    hw = raw_cfg.get("hardware", {}) or {}
    return HardwareConfig(
        device=str(hw.get("device", "auto")).lower(),
        gpu_id=int(hw.get("gpu_id", 0)),
        allow_gpu_batching=bool(hw.get("allow_gpu_batching", True)),
        relax_batch_size=int(hw.get("relax_batch_size", 8)),
        bandgap_batch_size=int(hw.get("bandgap_batch_size", 32)),
    )