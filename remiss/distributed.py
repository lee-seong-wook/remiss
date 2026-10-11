from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device | None = None
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: int | None = None


def is_distributed(context: DistributedContext | None) -> bool:
    return bool(context is not None and context.enabled and context.world_size > 1)


def is_main_process(context: DistributedContext | None) -> bool:
    return not is_distributed(context) or context.rank == 0


def setup_process_group(context: DistributedContext | None) -> None:
    if not is_distributed(context):
        return
    if context.device is None or context.device.type != "cuda":
        raise RuntimeError("Multi-GPU training requires CUDA devices.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if context.backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError(
            "Multi-GPU training uses DDP with the NCCL backend, but NCCL is not available. "
            "Run single-device training or use a PyTorch/Linux environment with NCCL support."
        )
    if dist.is_initialized():
        return
    if context.master_port is None:
        raise ValueError("DistributedContext.master_port is required for DDP setup.")
    dist.init_process_group(
        backend=context.backend,
        init_method=f"tcp://{context.master_addr}:{context.master_port}",
        rank=context.rank,
        world_size=context.world_size,
        device_id=context.device,
        timeout=timedelta(hours=1),
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext | None) -> None:
    if is_distributed(context):
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def all_gather_object(value: Any, context: DistributedContext | None) -> list[Any]:
    if not is_distributed(context):
        return [value]
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered
