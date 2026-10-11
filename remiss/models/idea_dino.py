from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch


def build_idea_dino_components(
    num_classes: int,
    *,
    dino_root: str | Path,
    config_file: str | Path,
    checkpoint: str | Path | None,
    finetune_ignore: Sequence[str],
    disable_torchvision_pretrained_backbone: bool,
) -> dict[str, object]:
    dino_root = Path(dino_root).resolve()
    if not dino_root.is_dir():
        raise FileNotFoundError(f"IDEA-DINO root does not exist: {dino_root}")
    config_path = (dino_root / config_file).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"File does not exist: {config_path}")

    args = _load_dino_args(dino_root, config_path, num_classes=int(num_classes))
    with _idea_dino_import_context(dino_root):
        try:
            dino_module = importlib.import_module("models.dino.dino")
        except ModuleNotFoundError as exc:
            if exc.name == "MultiScaleDeformableAttention":
                raise RuntimeError(
                    "IDEA-DINO requires its MultiScaleDeformableAttention extension. "
                    "Build it from third-party/DINO/models/dino/ops before using this backend."
                ) from exc
            raise
        if disable_torchvision_pretrained_backbone:
            backbone_module = importlib.import_module("models.dino.backbone")
            backbone_module.is_main_process = lambda: False
        model, criterion, postprocessors = dino_module.build_dino(args)

    if checkpoint is not None:
        _load_partial_checkpoint(model, checkpoint, ignore_keywords=finetune_ignore)

    return {"model": model, "criterion": criterion, "postprocessors": postprocessors}


def _load_dino_args(dino_root: Path, config_path: Path, *, num_classes: int) -> Namespace:
    with _idea_dino_import_context(dino_root):
        from util.slconfig import SLConfig

        config_dict = SLConfig.fromfile(str(config_path))._cfg_dict.to_dict()

    config_dict.update(
        {
            "config_file": str(config_path),
            "dataset_file": config_dict.get("dataset_file", "coco"),
            "device": "cpu",
            "distributed": False,
            "frozen_weights": config_dict.get("frozen_weights"),
            "num_classes": num_classes,
            "output_dir": "",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }
    )
    return Namespace(**config_dict)


def _load_partial_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    ignore_keywords: Sequence[str],
) -> None:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IDEA-DINO checkpoint does not exist: {path}")

    with torch.serialization.safe_globals([Namespace]):
        state = torch.load(path, map_location="cpu")["model"]

    model_state = model.state_dict()
    selected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        if any(token in key for token in ignore_keywords):
            continue
        if key not in model_state or not isinstance(value, torch.Tensor):
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            continue
        selected[key] = value
    model.load_state_dict(selected, strict=False)


@contextmanager
def _idea_dino_import_context(dino_root: Path):
    original_path = list(sys.path)
    ops_root = dino_root / "models" / "dino" / "ops"
    saved_models = sys.modules.get("models")
    saved_registry = sys.modules.get("models.registry")
    saved_util = sys.modules.get("util")
    saved_util_modules = {
        name: module for name, module in sys.modules.items() if name.startswith("util.")
    }
    saved_dino_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "models.dino" or name.startswith("models.dino.")
    }

    sys.modules.pop("models", None)
    sys.modules.pop("models.registry", None)
    sys.modules.pop("util", None)
    for name in saved_util_modules:
        sys.modules.pop(name, None)
    for name in saved_dino_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(ops_root))
    sys.path.insert(0, str(dino_root))
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "models.dino" or name.startswith("models.dino."):
                sys.modules.pop(name, None)
            if name == "util" or name.startswith("util."):
                sys.modules.pop(name, None)
        sys.modules.pop("models.registry", None)
        sys.modules.pop("models", None)
        if saved_models is not None:
            sys.modules["models"] = saved_models
        if saved_registry is not None:
            sys.modules["models.registry"] = saved_registry
        if saved_util is not None:
            sys.modules["util"] = saved_util
        sys.modules.update(saved_util_modules)
        sys.modules.update(saved_dino_modules)
