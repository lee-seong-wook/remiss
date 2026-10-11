from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remiss.config import load_runtime_config, resolve_device
from remiss.data import build_eval_dataloader
from remiss.engine import evaluate, persist_run_metadata
from remiss.module_configs import (
    resolve_module_config_paths,
    serialize_module_config_paths,
)
from remiss.module_metadata import collect_enabled_module_configs
from remiss.registry import build_model_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detection model from YAML configs.")
    parser.add_argument("--config", required=True, help="Path to the eval runtime YAML.")
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
        "--device",
        default=None,
        help="Optional override for runtime.device (for example: cpu, cuda, auto).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional override for checkpoint.path in the eval runtime YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config, runtime_config_path = load_runtime_config(
        args.config,
        mode="eval",
        dataset=args.data,
    )
    if args.output_dir:
        runtime_config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if args.device:
        runtime_config["device"] = args.device
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        runtime_config["checkpoint"]["path"] = str(checkpoint_path)
    elif not Path(runtime_config["checkpoint"]["path"]).is_file():
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {runtime_config['checkpoint']['path']}"
        )

    module_config_paths = resolve_module_config_paths(require_exists=False)
    runtime_config["_module_config_paths"] = serialize_module_config_paths(module_config_paths)

    model, model_config, arch, model_config_path = build_model_from_path(
        args.model,
        runtime_config=runtime_config,
        module_config_paths=module_config_paths,
    )
    device = resolve_device(runtime_config["device"])
    data_loader = build_eval_dataloader(runtime_config)
    output_dir = Path(runtime_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    persist_run_metadata(
        output_dir=output_dir,
        arch=arch,
        model_config=model_config,
        model_config_path=model_config_path,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        module_configs=collect_enabled_module_configs(
            arch,
            config_paths=module_config_paths,
        ),
    )

    print(f"Starting evaluation: arch={arch} device={device.type}")
    print(f"eval_config={runtime_config_path}")
    print(f"model_config={model_config_path}")
    if runtime_config.get("_dataset"):
        print(f"dataset={runtime_config['_dataset']}")
    print(f"checkpoint={runtime_config['checkpoint']['path']}")

    metrics, _ = evaluate(
        model=model,
        runtime_config=runtime_config,
        data_loader=data_loader,
        device=device,
        output_dir=output_dir,
        log_interval=runtime_config.get("eval", {}).get("log_interval", 20),
        stage_label="eval",
    )
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
