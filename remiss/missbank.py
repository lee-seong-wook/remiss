from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops


@dataclass(frozen=True, slots=True)
class MissBankMiningConfig:
    type: str
    start_epoch: int
    interval_epoch: int


@dataclass(frozen=True, slots=True)
class MissBankMatchingConfig:
    score_threshold: float | str
    iou_threshold: float | str


@dataclass(frozen=True, slots=True)
class MissBankTargetConfig:
    miss_threshold: int


@dataclass(frozen=True, slots=True)
class MissBankConfig:
    enabled: bool
    mining: MissBankMiningConfig
    matching: MissBankMatchingConfig
    target: MissBankTargetConfig
    arch: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, arch: str) -> MissBankConfig:
        merged = dict(raw)
        for key, value in raw.get("models", {}).get(arch, {}).items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        mining, matching, target = merged["mining"], merged["matching"], merged["target"]
        if mining["type"] != "offline":
            raise ValueError("MissBank supports only mining.type: offline.")
        return cls(
            enabled=bool(merged["enabled"]),
            mining=MissBankMiningConfig(
                type=mining["type"],
                start_epoch=int(mining["start_epoch"]),
                interval_epoch=int(mining["interval_epoch"]),
            ),
            matching=MissBankMatchingConfig(
                score_threshold=matching["score_threshold"],
                iou_threshold=matching["iou_threshold"],
            ),
            target=MissBankTargetConfig(miss_threshold=int(target["miss_threshold"])),
            arch=arch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mining": {
                "type": self.mining.type,
                "start_epoch": self.mining.start_epoch,
                "interval_epoch": self.mining.interval_epoch,
            },
            "matching": {
                "score_threshold": self.matching.score_threshold,
                "iou_threshold": self.matching.iou_threshold,
            },
            "target": {"miss_threshold": self.target.miss_threshold},
            "arch": self.arch,
        }


@dataclass(slots=True)
class MissBankRecord:
    record_key: str
    image_id: str
    gt_id: str
    gt_class: int
    bbox_xyxy: tuple[float, float, float, float]
    miss_count: int = 0
    is_missed: bool = False
    last_epoch: int = 0
    last_step: int = 0
    last_iou: float | None = None
    last_score: float | None = None
    best_iou: float | None = None
    best_iou_score: float | None = None
    best_score: float | None = None
    best_score_iou: float | None = None
    total_seen: int = 0
    total_missed: int = 0
    max_miss_count: int = 0
    consecutive_detect_count: int = 0

    def update(
        self,
        match: Mapping[str, Any],
        *,
        bbox_xyxy: tuple[float, float, float, float],
        epoch: int,
        step: int,
    ) -> None:
        self.bbox_xyxy = bbox_xyxy
        self.is_missed = not match["matched"]
        self.last_epoch = epoch
        self.last_step = step
        self.last_iou = match["iou"]
        self.last_score = match["score"]
        self.best_iou = match["best_iou"]
        self.best_iou_score = match["best_iou_score"]
        self.best_score = match["best_score"]
        self.best_score_iou = match["best_score_iou"]
        self.total_seen += 1
        if self.is_missed:
            self.miss_count += 1
            self.total_missed += 1
            self.max_miss_count = max(self.max_miss_count, self.miss_count)
            self.consecutive_detect_count = 0
        else:
            self.miss_count = 0
            self.consecutive_detect_count += 1

    def to_state(self) -> dict[str, Any]:
        return {**asdict(self), "bbox_xyxy": list(self.bbox_xyxy)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> MissBankRecord:
        return cls(**{**state, "bbox_xyxy": tuple(state["bbox_xyxy"])})


class MissBank(nn.Module):
    def __init__(self, config: MissBankConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._records: dict[str, MissBankRecord] = {}
        self._image_index: dict[str, set[str]] = defaultdict(set)
        self._stats: Counter[str] = Counter()

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def update(
        self,
        *,
        targets: Sequence[Mapping[str, Any]],
        detections: Sequence[Mapping[str, torch.Tensor]],
        epoch: int,
        step: int,
    ) -> None:
        for target, detection in zip(targets, detections, strict=True):
            self._stats.update(self._update_image(target, detection, int(epoch), int(step)))

    def get_records(self, image_id: Any) -> list[MissBankRecord]:
        return [self._records[key] for key in sorted(self._image_index.get(str(image_id), ()))]

    def epoch_snapshot(self, epoch: int) -> dict[str, Any]:
        epoch = int(epoch)
        threshold = self.config.target.miss_threshold
        records = [r for r in self._records.values() if r.last_epoch == epoch]
        missed = [r for r in records if r.is_missed]
        targets = [r for r in missed if r.miss_count >= threshold]
        miss_count_sum = sum(r.miss_count for r in missed)
        gt_miss_rate_sum = sum(r.total_missed / float(max(r.total_seen, 1)) for r in records)
        return {
            "epoch": epoch,
            "mining_type": self.config.mining.type,
            "miss_threshold": threshold,
            "num_seen_gts": len(records),
            "num_missed_gts": len(missed),
            "num_target_gts": len(targets),
            "num_images_seen": len({r.image_id for r in records}),
            "num_images_with_miss": len({r.image_id for r in missed}),
            "num_images_with_target": len({r.image_id for r in targets}),
            "miss_count_sum": miss_count_sum,
            "mean_miss_count": miss_count_sum / float(max(len(missed), 1)),
            "max_miss_count": max((r.miss_count for r in missed), default=0),
            "gt_miss_rate_sum": gt_miss_rate_sum,
            "mean_gt_miss_rate": gt_miss_rate_sum / float(max(len(records), 1)),
            "missed_gt_keys": sorted(r.record_key for r in missed),
            "target_gt_keys": sorted(r.record_key for r in targets),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "records": {key: record.to_state() for key, record in self._records.items()},
            "stats": dict(self._stats),
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.current_epoch = int(state["current_epoch"])
        self._records = {
            key: MissBankRecord.from_state(raw) for key, raw in state["records"].items()
        }
        self._image_index = defaultdict(set)
        for key, record in self._records.items():
            self._image_index[record.image_id].add(key)
        self._stats = Counter(state["stats"])

    def _update_image(
        self,
        target: Mapping[str, Any],
        detection: Mapping[str, torch.Tensor],
        epoch: int,
        step: int,
    ) -> Counter[str]:
        stats: Counter[str] = Counter()
        gt_boxes = target["boxes"].detach().cpu().to(dtype=torch.float32).reshape(-1, 4)
        if gt_boxes.numel() == 0:
            stats["images_without_gt"] += 1
            return stats

        image_id = str(target["image_id"].item())
        gt_labels = target["labels"].detach().cpu().to(dtype=torch.long).flatten()
        gt_ids = target["gt_ids"].tolist()
        pred_boxes = detection["boxes"].detach().to(dtype=torch.float32).reshape(-1, 4)
        pred_labels = detection["labels"].detach().to(dtype=torch.long).flatten()
        pred_scores = detection["scores"].detach().to(dtype=torch.float32).flatten()
        stats["images_seen"] += 1

        valid = (gt_boxes[:, 2] > gt_boxes[:, 0]) & (gt_boxes[:, 3] > gt_boxes[:, 1])
        num_invalid = int((~valid).sum().item())
        if num_invalid:
            stats["invalid_gt"] += num_invalid
        gt_indices = torch.where(valid)[0].tolist()
        gt_boxes = gt_boxes[valid]

        assignment, assign_stats = _assign_predictions(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels[valid],
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            score_threshold=float(self.config.matching.score_threshold),
            iou_threshold=float(self.config.matching.iou_threshold),
        )
        stats.update(assign_stats)

        for gt_index, gt_box, assigned_index in zip(gt_indices, gt_boxes, assignment, strict=True):
            gt_label = int(gt_labels[gt_index].item())
            bbox = tuple(gt_box.tolist())
            match = _match_gt(
                gt_box=gt_box,
                gt_label=gt_label,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                assigned_index=assigned_index,
            )
            record_key = f"{image_id}:ann:{gt_ids[gt_index]}"
            record = self._records.get(record_key)
            if record is None:
                record = MissBankRecord(
                    record_key=record_key,
                    image_id=image_id,
                    gt_id=str(gt_ids[gt_index]),
                    gt_class=gt_label,
                    bbox_xyxy=bbox,
                )
                self._records[record_key] = record
                self._image_index[image_id].add(record_key)
                stats["records_created"] += 1
            record.update(match, bbox_xyxy=bbox, epoch=epoch, step=step)
            stats["gt_seen"] += 1
            stats["gt_missed" if record.is_missed else "gt_detected"] += 1
        return stats


def load_missbank_config(path: str | Path, *, arch: str) -> MissBankConfig:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        return MissBankConfig.from_mapping(yaml.safe_load(handle), arch=arch)


def build_missbank_from_yaml(
    path: str | Path,
    *,
    arch: str,
    detector_score_threshold: float,
    detector_iou_threshold: float,
) -> MissBank | None:
    config = load_missbank_config(path, arch=arch)
    if not config.enabled:
        return None
    matching = MissBankMatchingConfig(
        score_threshold=_resolve_threshold(
            config.matching.score_threshold, detector_score_threshold
        ),
        iou_threshold=_resolve_threshold(config.matching.iou_threshold, detector_iou_threshold),
    )
    return MissBank(replace(config, matching=matching))


def compute_missbank_stability_metrics(
    current_snapshot: Mapping[str, Any],
    *,
    previous_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_missed = set(current_snapshot["missed_gt_keys"])
    num_missed = len(current_missed)
    num_target = len(set(current_snapshot["target_gt_keys"]))

    jaccard = churn_rate = new_miss_rate = None
    if previous_snapshot is not None:
        previous_missed = set(previous_snapshot["missed_gt_keys"])
        union = current_missed | previous_missed
        jaccard = len(current_missed & previous_missed) / float(len(union)) if union else 1.0
        churn_rate = 1.0 - jaccard
        new_miss_rate = (
            len(current_missed - previous_missed) / float(num_missed) if num_missed else 0.0
        )

    return {
        "epoch": current_snapshot["epoch"],
        "num_seen_gts": current_snapshot["num_seen_gts"],
        "num_missed_gts": num_missed,
        "num_target_gts": num_target,
        "num_images_seen": current_snapshot["num_images_seen"],
        "num_images_with_miss": current_snapshot["num_images_with_miss"],
        "num_images_with_target": current_snapshot["num_images_with_target"],
        "miss_gt_jaccard_stability": jaccard,
        "miss_gt_churn_rate": churn_rate,
        "new_miss_rate": new_miss_rate,
        "persistent_miss_ratio": num_target / float(num_missed) if num_missed else None,
    }


def merge_missbank_epoch_snapshots(
    snapshots: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    valid = [snapshot for snapshot in snapshots if snapshot is not None]
    if not valid:
        return None

    count_keys = (
        "num_seen_gts",
        "num_images_seen",
        "num_images_with_miss",
        "num_images_with_target",
    )
    merged: dict[str, Any] = {
        "epoch": max(snapshot["epoch"] for snapshot in valid),
        "mining_type": valid[0]["mining_type"],
        "miss_threshold": valid[0]["miss_threshold"],
        **{key: sum(snapshot[key] for snapshot in valid) for key in count_keys},
    }
    missed_gt_keys = set().union(*(snapshot["missed_gt_keys"] for snapshot in valid))
    target_gt_keys = set().union(*(snapshot["target_gt_keys"] for snapshot in valid))
    miss_count_sum = sum(snapshot["miss_count_sum"] for snapshot in valid)
    gt_miss_rate_sum = 0.0
    for snapshot in valid:
        gt_miss_rate_sum += float(snapshot["gt_miss_rate_sum"])
    merged.update(
        {
            "num_missed_gts": len(missed_gt_keys),
            "num_target_gts": len(target_gt_keys),
            "miss_count_sum": miss_count_sum,
            "mean_miss_count": miss_count_sum / float(max(len(missed_gt_keys), 1)),
            "max_miss_count": max(snapshot["max_miss_count"] for snapshot in valid),
            "gt_miss_rate_sum": gt_miss_rate_sum,
            "mean_gt_miss_rate": gt_miss_rate_sum / float(max(merged["num_seen_gts"], 1)),
            "missed_gt_keys": sorted(missed_gt_keys),
            "target_gt_keys": sorted(target_gt_keys),
        }
    )
    return merged


def _assign_predictions(
    *,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[list[int], Counter[str]]:
    assignment = [-1] * int(gt_boxes.shape[0])
    stats: Counter[str] = Counter()
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return assignment, stats

    ious = box_ops.box_iou(gt_boxes, pred_boxes.cpu()).clamp(min=0.0, max=1.0)
    eligible = (
        (gt_labels.reshape(-1, 1) == pred_labels.cpu().reshape(1, -1))
        & (pred_scores.cpu().reshape(1, -1) >= score_threshold)
        & (ious >= iou_threshold)
    )
    for gt_index in range(len(assignment)):
        candidates = torch.where(eligible[gt_index])[0]
        if candidates.numel() > 0:
            best = int(torch.argmax(ious[gt_index, candidates]).item())
            assignment[gt_index] = int(candidates[best].item())

    shared = {index for index, count in Counter(assignment).items() if index >= 0 and count > 1}
    if shared:
        stats["shared_preds"] += len(shared)
        for gt_index, pred_index in enumerate(assignment):
            if pred_index in shared:
                stats["duplicate_matched_gts"] += 1
                stats[f"duplicate_gt_class_{int(gt_labels[gt_index].item())}"] += 1
    stats["gt_assigned"] += sum(1 for index in assignment if index >= 0)
    return assignment, stats


def _match_gt(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    assigned_index: int,
) -> dict[str, Any]:
    match: dict[str, Any] = {
        "matched": False,
        "iou": None,
        "score": None,
        "best_iou": None,
        "best_iou_score": None,
        "best_score": None,
        "best_score_iou": None,
    }
    if pred_boxes.numel() == 0:
        return match
    gt_box = gt_box.to(device=pred_boxes.device).reshape(1, 4)
    ious = box_ops.box_iou(gt_box, pred_boxes)[0].clamp(min=0.0, max=1.0)
    indices = torch.where(pred_labels == gt_label)[0]
    if indices.numel() > 0:
        best_iou_index = int(indices[int(torch.argmax(ious[indices]).item())].item())
        best_score_index = int(indices[int(torch.argmax(pred_scores[indices]).item())].item())
        match["best_iou"] = float(ious[best_iou_index].item())
        match["best_iou_score"] = float(pred_scores[best_iou_index].item())
        match["best_score"] = float(pred_scores[best_score_index].item())
        match["best_score_iou"] = float(ious[best_score_index].item())
    if assigned_index >= 0:
        match["matched"] = True
        match["iou"] = float(ious[assigned_index].item())
        match["score"] = float(pred_scores[assigned_index].item())
    return match


def _resolve_threshold(value: float | str, detector_value: float) -> float:
    if isinstance(value, str) and value.lower() == "auto":
        return float(detector_value)
    return float(value)
