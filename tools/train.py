from __future__ import annotations

import argparse
import socket
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remiss.config import format_device_name, load_runtime_config, resolve_devices
from remiss.data import build_train_dataloaders
from remiss.distributed import (
    DistributedContext,
    cleanup_process_group,
    setup_process_group,
)
from remiss.engine import fit, seed_everything
from remiss.module_configs import (
    resolve_module_config_paths,
    serialize_module_config_paths,
)
from remiss.registry import build_model_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detection model from YAML configs.")
    parser.add_argument("--config", required=True, help="Path to the train runtime YAML.")
    parser.add_argument("--model", required=True, help="Path to the model YAML.")
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Dataset name used to select <DATASET>_* data path env vars "
            "(kitti, bdd100k, cityscapes, nuimages)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for runtime.output_dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for runtime.seed.",
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=None,
        help=(
            "Optional override for runtime.device. Use one value for single-device "
            "training or multiple CUDA values for DDP, for example: cuda:0 cuda:1."
        ),
    )
    parser.add_argument(
        "--missbank-config",
        default=None,
        help="Optional override for the MissBank YAML config path.",
    )
    parser.add_argument(
        "--hard-replay-config",
        default=None,
        help="Optional override for the Hard Replay YAML config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config, runtime_config_path = load_runtime_config(
        args.config,
        mode="train",
        dataset=args.data,
    )
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be >= 0.")
        runtime_config["seed"] = int(args.seed)
    if args.output_dir:
        runtime_config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        runtime_config["checkpoint"]["dir"] = str(
            (Path(runtime_config["output_dir"]) / "checkpoints").resolve()
        )
    if args.device:
        runtime_config["device"] = args.device[0] if len(args.device) == 1 else list(args.device)

    overrides = {"missbank": args.missbank_config, "hard_replay": args.hard_replay_config}
    module_config_paths = resolve_module_config_paths(overrides, require_exists=False)
    missing = [
        f"{name}={module_config_paths[name]}"
        for name, raw_path in overrides.items()
        if raw_path is not None and not module_config_paths[name].is_file()
    ]
    if missing:
        raise FileNotFoundError("Module config override file was not found: " + ", ".join(missing))
    override_names = [name for name, raw_path in overrides.items() if raw_path is not None]
    runtime_config["_module_config_paths"] = serialize_module_config_paths(module_config_paths)
    runtime_config["_module_config_overrides"] = override_names

    devices = resolve_devices(runtime_config["device"])
    device_names = [format_device_name(device) for device in devices]
    runtime_config["device"] = device_names[0]
    runtime_config["devices"] = device_names

    if len(devices) > 1:
        _run_distributed_training(
            model_path=args.model,
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            device_names=device_names,
        )
        return

    device = devices[0]
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    seed_everything(runtime_config["seed"])
    model, model_config, arch, model_config_path = build_model_from_path(
        args.model,
        runtime_config=runtime_config,
        module_config_paths=module_config_paths,
    )
    train_loader, val_loader = build_train_dataloaders(
        runtime_config,
        arch=arch,
        module_config_paths=module_config_paths,
    )

    print(f"Starting training: arch={arch} device={format_device_name(device)}")
    print(f"train_config={runtime_config_path}")
    print(f"model_config={model_config_path}")
    print(f"seed={runtime_config['seed']}")
    if runtime_config.get("_dataset"):
        print(f"dataset={runtime_config['_dataset']}")
    _print_module_config_overrides(module_config_paths, override_names)

    fit(
        model=model,
        model_config=model_config,
        model_config_path=model_config_path,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        arch=arch,
        module_config_paths=module_config_paths,
    )


def _run_distributed_training(
    *,
    model_path: str,
    runtime_config: dict,
    runtime_config_path: Path,
    device_names: list[str],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    is_nccl_available = getattr(torch.distributed, "is_nccl_available", lambda: False)
    if not is_nccl_available():
        raise RuntimeError(
            "Multi-GPU training uses DDP with NCCL, but NCCL is not available. "
            "Use single-device training or a PyTorch/Linux environment with NCCL support."
        )
    master_port = _find_free_port()
    mp.spawn(
        _distributed_train_worker,
        args=(
            device_names,
            model_path,
            runtime_config,
            runtime_config_path,
            "127.0.0.1",
            master_port,
        ),
        nprocs=len(device_names),
        join=True,
    )


def _distributed_train_worker(
    rank: int,
    device_names: list[str],
    model_path: str,
    runtime_config: dict,
    runtime_config_path: Path,
    master_addr: str,
    master_port: int,
) -> None:
    local_config = deepcopy(runtime_config)
    devices = [torch.device(name) for name in device_names]
    device = devices[rank]
    if device.index is None:
        raise RuntimeError("DDP workers require explicit CUDA device indices.")
    torch.cuda.set_device(device.index)
    context = DistributedContext(
        enabled=True,
        rank=rank,
        world_size=len(devices),
        local_rank=rank,
        device=device,
        master_addr=master_addr,
        master_port=master_port,
    )
    module_config_paths = local_config["_module_config_paths"]
    setup_process_group(context)
    try:
        local_config["device"] = device_names[rank]
        local_config["devices"] = list(device_names)
        seed_everything(int(local_config["seed"]))
        model, model_config, arch, model_config_path = build_model_from_path(
            model_path,
            runtime_config=local_config,
            module_config_paths=module_config_paths,
        )
        train_loader, val_loader = build_train_dataloaders(
            local_config,
            arch=arch,
            distributed=True,
            rank=rank,
            world_size=len(devices),
            module_config_paths=module_config_paths,
        )

        if rank == 0:
            print(f"Starting DDP training: arch={arch} devices={', '.join(device_names)}")
            print(f"train_config={runtime_config_path}")
            print(f"model_config={model_config_path}")
            print(f"seed={local_config['seed']}")
            if local_config.get("_dataset"):
                print(f"dataset={local_config['_dataset']}")
            _print_module_config_overrides(
                module_config_paths, local_config["_module_config_overrides"]
            )

        fit(
            model=model,
            model_config=model_config,
            model_config_path=model_config_path,
            runtime_config=local_config,
            runtime_config_path=runtime_config_path,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            arch=arch,
            distributed=context,
            module_config_paths=module_config_paths,
        )
    finally:
        cleanup_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _print_module_config_overrides(
    module_config_paths: dict[str, str | Path],
    override_names: list[str],
) -> None:
    if not module_config_paths or not override_names:
        return
    print("module_configs:")
    for name in ("missbank", "hard_replay"):
        path = module_config_paths.get(name)
        if name in override_names and path is not None:
            print(f"  {name}: {Path(path)}")


if __name__ == "__main__":
    main()
