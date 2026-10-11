from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any

import torch
from pycocotools.cocoeval import COCOeval

BOX_METRIC_NAMES = (
    "bbox_mAP_50_95",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_mAR_1",
    "bbox_mAR_10",
    "bbox_mAR_100",
    "bbox_mAR_small",
    "bbox_mAR_medium",
    "bbox_mAR_large",
)


def evaluate_detection(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    metrics_type: str,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if metrics_type != "coco_detection":
        raise ValueError(f"Unsupported metrics.type: {metrics_type!r}")

    dataset = data_loader.dataset

    model_was_training = model.training
    model.eval()

    predictions: list[dict[str, Any]] = []
    total_steps = len(data_loader)
    eta_label = "epoch_eta" if epoch_index is not None and total_epochs is not None else "eval_eta"
    start_time = time.perf_counter()
    processed_images = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                outputs = model(images)

            for target, output in zip(targets, outputs):
                predictions.extend(_to_coco_predictions(int(target["image_id"].item()), output))

            processed_images += len(images)

            if log_interval > 0 and (step % log_interval == 0 or step == total_steps):
                elapsed = time.perf_counter() - start_time
                prefix = _build_progress_prefix(
                    stage_label=stage_label,
                    step=step,
                    total_steps=total_steps,
                    eta_label=eta_label,
                    eta_seconds=elapsed / step * max(total_steps - step, 0),
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                )
                print(f"{prefix} images={processed_images} predictions={len(predictions)}")

    metrics = _compute_box_metrics(dataset.coco, predictions)
    metrics["num_images"] = float(len(dataset))
    metrics["num_predictions"] = float(len(predictions))

    if model_was_training:
        model.train()

    return metrics, predictions


def save_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def _to_coco_predictions(
    image_id: int,
    output: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    results: list[dict[str, Any]] = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(max(0.0, x2 - x1)),
                    float(max(0.0, y2 - y1)),
                ],
                "score": float(score.item()),
            }
        )
    return results


def _compute_box_metrics(coco_gt, predictions: list[dict[str, Any]]) -> dict[str, float]:
    if not predictions:
        return {name: 0.0 for name in BOX_METRIC_NAMES}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    return {
        name: float(value) for name, value in zip(BOX_METRIC_NAMES, evaluator.stats, strict=True)
    }


def _build_progress_prefix(
    stage_label: str,
    step: int,
    total_steps: int,
    eta_label: str,
    eta_seconds: float,
    epoch_index: int | None,
    total_epochs: int | None,
) -> str:
    parts = [f"[{stage_label}]"]
    if epoch_index is not None and total_epochs is not None:
        parts.append(f"epoch {epoch_index + 1}/{total_epochs}")
    parts.append(f"step {step}/{total_steps}")
    parts.append(f"{eta_label}={_format_eta(eta_seconds)}")
    return " ".join(parts)


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
