from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.nn as nn

from .arch import normalize_arch
from .config import load_yaml_file
from .dataset_meta import infer_num_classes_from_runtime_config
from .missbank import build_missbank_from_yaml
from .models import DINOWrapper, FasterRCNNWrapper, FCOSWrapper

MODEL_BUILDERS = {
    "dino": DINOWrapper,
    "fasterrcnn": FasterRCNNWrapper,
    "fcos": FCOSWrapper,
}

DETECTOR_THRESHOLD_KEYS = {
    "fcos": ("head", "score_thresh", 0.2, "nms_thresh", 0.6),
    "fasterrcnn": ("roi_head", "box_score_thresh", 0.05, "box_nms_thresh", 0.5),
    "dino": ("inference", "score_thresh", 0.0, "nms_thresh", 0.5),
}


def build_model_from_path(
    model_config_path: str | Path,
    runtime_config: dict[str, Any] | None = None,
    *,
    module_config_paths: Mapping[str, str | Path] | None = None,
) -> tuple[nn.Module, dict[str, Any], str, Path]:
    resolved_path = Path(model_config_path).expanduser().resolve()
    model_config = load_yaml_file(resolved_path)
    if runtime_config is not None:
        model_config = dict(model_config)
        model_config["num_classes"] = infer_num_classes_from_runtime_config(runtime_config)

    arch = normalize_arch(model_config.get("arch") or resolved_path.stem)
    builder = MODEL_BUILDERS.get(arch)
    if builder is None:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise NotImplementedError(
            f"Model arch {arch!r} is not implemented. Supported arches: {supported}. "
            "If your YAML filename does not match the arch name, add an explicit 'arch:' field."
        )
    model = builder(model_config)

    missbank_path = (module_config_paths or {}).get("missbank")
    if missbank_path is not None:
        score_threshold, iou_threshold = _detector_thresholds(model_config, arch)
        missbank = build_missbank_from_yaml(
            missbank_path,
            arch=arch,
            detector_score_threshold=score_threshold,
            detector_iou_threshold=iou_threshold,
        )
        if missbank is not None:
            model.missbank = missbank
    return model, model_config, arch, resolved_path


def _detector_thresholds(model_config: Mapping[str, Any], arch: str) -> tuple[float, float]:
    section_name, score_key, score_default, iou_key, iou_default = DETECTOR_THRESHOLD_KEYS[arch]
    section = model_config.get(section_name, {})
    return float(section.get(score_key, score_default)), float(section.get(iou_key, iou_default))
