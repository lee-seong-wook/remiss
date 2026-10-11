from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import torch
from torchvision.ops import nms

from .base import BaseDetectionWrapper


class DINOWrapper(BaseDetectionWrapper):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.num_classes = int(cfg["num_classes"])
        self.inference_cfg = cfg.get("inference", {})
        self.loss_cfg = cfg.get("loss", {})

        backend_cfg = cfg["backend"]
        module_name, _, builder_name = backend_cfg["builder"].partition(":")
        builder = getattr(importlib.import_module(module_name), builder_name)
        built = builder(num_classes=self.num_classes, **backend_cfg.get("kwargs", {}))

        self.model = built["model"]
        self.criterion = built["criterion"]
        self.bbox_postprocessor = built["postprocessors"]["bbox"]

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        original_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
        if targets is None:
            prepared_targets = None
            outputs = self.model(images)
        else:
            prepared_targets = self._prepare_targets(targets, original_sizes)
            outputs = self.model(images, prepared_targets)

        if self.training:
            return self._compute_losses(outputs, prepared_targets)

        target_sizes = torch.tensor(
            original_sizes, dtype=torch.int64, device=outputs["pred_logits"].device
        )
        predictions = self.bbox_postprocessor(outputs, target_sizes)
        return [self._finalize_prediction(prediction) for prediction in predictions]

    def _compute_losses(
        self,
        outputs: dict[str, Any],
        prepared_targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        weighted_losses: dict[str, torch.Tensor] = {}
        for name, value in self.criterion(outputs, prepared_targets).items():
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                continue
            weight = self._loss_weight(name)
            if weight is not None:
                weighted_losses[name] = value * weight
        return weighted_losses

    def _loss_weight(self, name: str) -> float | None:
        lower_name = name.lower()
        if "class_error" in lower_name or "cardinality" in lower_name:
            return None
        if "loss" not in lower_name:
            return None
        if "bbox" in lower_name:
            return float(self.loss_cfg.get("bbox_weight", 5.0))
        if "giou" in lower_name:
            return float(self.loss_cfg.get("giou_weight", 2.0))
        if "ce" in lower_name or "cls" in lower_name or "class" in lower_name:
            return float(self.loss_cfg.get("class_weight", 1.0))
        return 1.0

    def _finalize_prediction(self, prediction: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        boxes = prediction["boxes"].to(dtype=torch.float32)
        scores = prediction["scores"].to(dtype=torch.float32)
        labels = prediction["labels"].to(dtype=torch.int64) + 1

        keep = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        score_thresh = float(self.inference_cfg.get("score_thresh", 0.0))
        if score_thresh > 0.0:
            keep &= scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        topk = self.inference_cfg.get("topk")
        if topk is not None and int(topk) > 0 and scores.numel() > int(topk):
            topk_indices = torch.topk(scores, k=int(topk)).indices
            boxes = boxes[topk_indices]
            scores = scores[topk_indices]
            labels = labels[topk_indices]

        if bool(self.inference_cfg.get("use_nms", False)) and boxes.numel() > 0:
            keep_indices = nms(boxes, scores, float(self.inference_cfg.get("nms_thresh", 0.5)))
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        return {"boxes": boxes, "scores": scores, "labels": labels}

    def _prepare_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        prepared = []
        for target, (height, width) in zip(targets, original_sizes, strict=True):
            boxes_xyxy = target["boxes"]
            labels_raw = target["labels"].to(dtype=torch.int64)
            labels = labels_raw - 1

            if labels.numel() > 0:
                min_label = int(labels.min().item())
                max_label = int(labels.max().item())
                if min_label < 0 or max_label >= self.num_classes:
                    raise ValueError(
                        "DINO labels must map into [0, num_classes). "
                        f"Received range [{min_label}, {max_label}] for num_classes={self.num_classes}."
                    )

            size_tensor = torch.tensor([height, width], dtype=torch.int64, device=boxes_xyxy.device)
            prepared_target = {
                "boxes": _xyxy_to_normalized_cxcywh(boxes_xyxy, width=width, height=height),
                "boxes_xyxy": boxes_xyxy.to(dtype=torch.float32),
                "labels": labels,
                "labels_raw": labels_raw,
                "image_id": target["image_id"].to(dtype=torch.int64),
                "size": size_tensor,
                "orig_size": size_tensor.clone(),
            }
            if "area" in target:
                prepared_target["area"] = target["area"].to(dtype=torch.float32)
            if "iscrowd" in target:
                prepared_target["iscrowd"] = target["iscrowd"].to(dtype=torch.int64)
            prepared.append(prepared_target)
        return prepared


def _xyxy_to_normalized_cxcywh(boxes_xyxy: torch.Tensor, *, width: int, height: int):
    boxes = boxes_xyxy.to(dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes / boxes.new_tensor([width, height, width, height])
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack((cx, cy, w, h), dim=-1)
