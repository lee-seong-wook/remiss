from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import Sampler

from .arch import normalize_arch


@dataclass(frozen=True, slots=True)
class HardReplayConfig:
    enabled: bool = False
    start_epoch: int = 2
    replay_ratio: float = 0.25
    beta: float = 1.0
    temperature: float = 1.0
    max_image_weight: float = 5.0
    min_image_weight: float = 1.0
    max_replays_per_gt_per_epoch: int = 4

    def __post_init__(self) -> None:
        for invalid, message in (
            (self.start_epoch < 1, "start_epoch must be >= 1"),
            (not 0.0 < self.replay_ratio < 1.0, "replay_ratio must satisfy 0 < value < 1"),
            (self.beta < 0.0, "beta must be >= 0"),
            (self.temperature <= 0.0, "temperature must be > 0"),
            (self.min_image_weight <= 0.0, "min_image_weight must be > 0"),
            (
                self.min_image_weight > self.max_image_weight,
                "min_image_weight must be <= max_image_weight",
            ),
            (self.max_replays_per_gt_per_epoch < 1, "max_replays_per_gt_per_epoch must be >= 1"),
        ):
            if invalid:
                raise ValueError(f"Hard Replay {message}.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, arch: str | None = None) -> HardReplayConfig:
        data = dict(raw)
        arch = normalize_arch(arch)
        if arch is not None:
            data.update(data.get("models", {}).get(arch) or {})
        return cls(**{f.name: type(f.default)(data[f.name]) for f in fields(cls) if f.name in data})


@dataclass(frozen=True, slots=True)
class HardReplaySampleRef:
    dataset_index: int
    hard_replay: bool = True


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    dataset_index: int
    weight: float
    cap: int
    num_gt: int


@dataclass(slots=True)
class ReplayIndex:
    summary: dict[str, Any]
    candidates: list[ReplayCandidate] = field(default_factory=list)


class MixedReplayBatchSampler(Sampler[list[Any]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        replay_ratio: float,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.base_only = False
        self._index = ReplayIndex(_summary(epoch=0, reason="inactive"))
        self._last_summary = dict(self._index.summary)

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_base_only(self, enabled: bool) -> None:
        self.base_only = bool(enabled)

    def set_replay_index(self, replay_index: ReplayIndex, *, epoch: int) -> None:
        self._index = replay_index
        self.set_epoch(epoch)
        num_base = self._num_base_samples()
        planned = self._planned_replay_count(num_base)
        counts = []
        if planned > 0:
            counts = self._replay_counts(planned, math.ceil((num_base + planned) / self.batch_size))
        slots = max(counts, default=0)
        self._last_summary = {
            **replay_index.summary,
            "replay_budget_requested": self._replay_request(num_base),
            "replay_budget_planned": planned,
            "replay_slots_per_batch": slots,
            "base_slots_per_batch": self.batch_size - slots,
        }

    def __len__(self) -> int:
        num_base = self._num_base_samples()
        return math.ceil((num_base + self._planned_replay_count(num_base)) / self.batch_size)

    def __iter__(self):
        base = self._base_indices()
        num_base = len(base)
        requested = self._replay_request(num_base)
        planned = self._planned_replay_count(num_base)
        schedule: list[int] = []
        if planned > 0:
            schedule = self._replay_schedule(planned)
            if self.world_size > 1 and dist.is_initialized():
                count = torch.tensor([len(schedule)], dtype=torch.long)
                dist.all_reduce(count, op=dist.ReduceOp.MIN)
                schedule = schedule[: int(count.item())]
        if not schedule:
            for start in range(0, num_base, self.batch_size):
                yield base[start : start + self.batch_size]
            if not self.base_only:
                self._finalize_summary(num_base, requested, [], [])
            return

        counts = self._replay_counts(
            len(schedule), math.ceil((num_base + len(schedule)) / self.batch_size)
        )
        replay_cursor = base_cursor = 0
        for batch_number, count in enumerate(counts, start=1):
            replay_slice = schedule[replay_cursor : replay_cursor + count]
            base_batch = base[base_cursor : base_cursor + self.batch_size - count]
            replay_cursor += count
            base_cursor += len(base_batch)
            batch = [*base_batch, *map(HardReplaySampleRef, replay_slice)]
            random.Random(self.seed + self.epoch * 1009 + batch_number).shuffle(batch)
            yield batch
        self._finalize_summary(num_base, requested, schedule, counts)

    def _num_base_samples(self) -> int:
        return math.ceil(self.dataset_size / self.world_size)

    def _base_indices(self) -> list[int]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        needed = self._num_base_samples() * self.world_size - len(indices)
        if needed > 0:
            indices += (indices * math.ceil(needed / len(indices)))[:needed]
        return indices[self.rank :: self.world_size]

    def _replay_request(self, num_base: int) -> int:
        if self.base_only or not self._index.candidates:
            return 0
        return math.floor(num_base * self.replay_ratio)

    def _planned_replay_count(self, num_base: int) -> int:
        if self.batch_size < 2:
            return 0
        capacity = sum(candidate.cap for candidate in self._index.candidates)
        return min(self._replay_request(num_base), capacity)

    def _replay_counts(self, total: int, num_batches: int) -> list[int]:
        counts = [total // num_batches] * num_batches
        remainder = total % num_batches
        if remainder:
            positions = list(range(num_batches))
            random.Random(self.seed + self.epoch * 104729 + self.rank * 65537 + 31).shuffle(
                positions
            )
            for position in positions[:remainder]:
                counts[position] += 1
        return counts

    def _replay_schedule(self, num_samples: int) -> list[int]:
        indices: list[int] = []
        weights: list[float] = []
        for candidate in self._index.candidates:
            indices += [candidate.dataset_index] * candidate.cap
            weights += [candidate.weight] * candidate.cap
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1000003 + self.rank * 9176 + 17)
        selected = torch.multinomial(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=num_samples,
            replacement=False,
            generator=generator,
        )
        return [indices[i] for i in selected.tolist()]

    def _finalize_summary(
        self,
        num_base: int,
        requested: int,
        replayed: Sequence[int],
        counts: Sequence[int],
    ) -> None:
        summary = self._index.summary
        num_gt = {c.dataset_index: c.num_gt for c in self._index.candidates}
        exposure = sum(num_gt[index] for index in replayed)
        num_active_gt = summary["replay_num_active_gt"]
        num_replayed = len(replayed)
        total = num_base + num_replayed
        slots = max(counts, default=0)
        self._last_summary = {
            **summary,
            "active": num_replayed > 0,
            "replay_ratio_effective": num_replayed / num_base if num_base > 0 else 0.0,
            "replay_ratio_effective_total": num_replayed / total if total > 0 else 0.0,
            "replay_budget_requested": requested,
            "replay_budget_planned": self._planned_replay_count(num_base),
            "replay_budget_actual": num_replayed,
            "replay_samples": num_replayed,
            "replay_unique_images": len(set(replayed)),
            "replay_exposure_per_gt": exposure / num_active_gt if num_active_gt > 0 else 0.0,
            "replay_batches": sum(1 for count in counts if count > 0),
            "replay_slots_per_batch": slots,
            "base_slots_per_batch": self.batch_size - slots,
            "base_samples": num_base,
            "base_samples_original": num_base,
        }


class HardReplayController:
    def __init__(
        self,
        *,
        config: HardReplayConfig,
        dataset: Any,
        batch_sampler: MixedReplayBatchSampler,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def refresh(self, *, missbank: Any, epoch: int) -> None:
        epoch = int(epoch)
        self.batch_sampler.set_replay_index(self._build_index(missbank, epoch), epoch=epoch)

    def set_base_only(self, enabled: bool) -> None:
        self.batch_sampler.set_base_only(enabled)

    def summary(self) -> dict[str, Any]:
        return self.batch_sampler.last_summary

    def _build_index(self, missbank: Any, epoch: int) -> ReplayIndex:
        cfg = self.config
        if epoch < cfg.start_epoch:
            return ReplayIndex(_summary(epoch=epoch, reason="warmup"))

        records_per_image = [missbank.get_records(image_id) for image_id in self.dataset.image_ids]
        latest_mined_epoch = max(
            (record.last_epoch for records in records_per_image for record in records),
            default=None,
        )
        epochs_since_mining = None if latest_mined_epoch is None else epoch - latest_mined_epoch
        score_threshold = float(missbank.config.matching.score_threshold)
        iou_threshold = float(missbank.config.matching.iou_threshold)

        candidates: list[ReplayCandidate] = []
        image_weights: list[float] = []
        gt_keys: set[str] = set()
        priority_sum = 0.0
        for dataset_index, records in enumerate(records_per_image):
            records = [
                record
                for record in records
                if record.is_missed and record.last_epoch == latest_mined_epoch
            ]
            if not records:
                continue
            priorities = [_record_priority(r, score_threshold, iou_threshold) for r in records]
            priority = sum(sorted(priorities, reverse=True))
            weight = max(cfg.min_image_weight, min(cfg.max_image_weight, 1.0 + cfg.beta * priority))
            candidates.append(
                ReplayCandidate(
                    dataset_index=dataset_index,
                    weight=weight**cfg.temperature,
                    cap=cfg.max_replays_per_gt_per_epoch * len(records),
                    num_gt=len(records),
                )
            )
            image_weights.append(weight)
            gt_keys.update(record.record_key for record in records)
            priority_sum += priority

        summary = _summary(
            epoch=epoch,
            reason="active" if candidates else "no_missed_gt",
            replay_ratio_requested=cfg.replay_ratio,
            latest_mined_epoch=latest_mined_epoch,
            epochs_since_mining=epochs_since_mining,
        )
        if candidates:
            summary.update(
                replay_num_images=len(candidates),
                replay_num_active_gt=len(gt_keys),
                replay_mean_image_weight=sum(image_weights) / len(image_weights),
                replay_mean_gt_priority=priority_sum / len(gt_keys),
            )
        return ReplayIndex(summary, candidates)


def load_hard_replay_config(path: str | Path, *, arch: str | None = None) -> HardReplayConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Hard Replay YAML must contain a mapping at the top level: {config_path}")
    return HardReplayConfig.from_mapping(raw, arch=arch)


def build_hard_replay_controller_from_yaml(
    path: str | Path,
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    arch: str | None = None,
) -> HardReplayController | None:
    config = load_hard_replay_config(path, arch=arch)
    if not config.enabled:
        return None
    batch_sampler = MixedReplayBatchSampler(
        dataset_size=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        replay_ratio=config.replay_ratio,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    return HardReplayController(config=config, dataset=dataset, batch_sampler=batch_sampler)


def _summary(*, epoch: int, reason: str, **updates: Any) -> dict[str, Any]:
    summary = {
        "enabled": True,
        "active": False,
        "epoch": epoch,
        "reason": reason,
        "replay_ratio_requested": 0.0,
        "replay_ratio_policy": "epoch",
        "replay_ratio_effective": 0.0,
        "replay_ratio_effective_total": 0.0,
        "replay_budget_requested": 0,
        "replay_budget_planned": 0,
        "replay_budget_actual": 0,
        "replay_num_images": 0,
        "replay_num_active_gt": 0,
        "latest_mined_epoch": None,
        "epochs_since_mining": None,
        "replay_mean_image_weight": 0.0,
        "replay_mean_gt_priority": 0.0,
        "replay_samples": 0,
        "replay_unique_images": 0,
        "replay_exposure_per_gt": 0.0,
        "replay_batches": 0,
        "replay_slots_per_batch": 0,
        "base_slots_per_batch": 0,
        "base_samples": 0,
        "base_samples_original": 0,
        "epoch_size_policy": "expand",
    }
    summary.update(updates)
    return summary


def _record_priority(record: Any, score_threshold: float, iou_threshold: float) -> float:
    iou_gap = 0.0 if record.best_iou is None else max(0.0, iou_threshold - record.best_iou)
    score_gap = 0.0 if record.best_score is None else max(0.0, score_threshold - record.best_score)
    return (
        record.miss_count
        + 0.25 * record.total_missed
        + 0.1 * record.max_miss_count
        + iou_gap
        + 0.25 * score_gap
    )
