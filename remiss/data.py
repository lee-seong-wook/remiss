from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F

from .hard_replay import HardReplaySampleRef, build_hard_replay_controller_from_yaml


class CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    def __init__(self, images_dir: str | Path, annotations_path: str | Path) -> None:
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.coco = COCO(str(Path(annotations_path).expanduser().resolve()))
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int | HardReplaySampleRef) -> tuple[torch.Tensor, dict[str, Any]]:
        replay = isinstance(index, HardReplaySampleRef)
        image_id = self.image_ids[int(index.dataset_index if replay else index)]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        width, height = image.size
        target = _build_target(int(image_id), width, height, annotations)
        if replay and index.hard_replay:
            target["hard_replay"] = torch.tensor(True, dtype=torch.bool)
        return F.to_tensor(image), target


def build_train_dataloaders(
    config: dict[str, Any],
    *,
    arch: str | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    module_config_paths: dict[str, str | Path] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    data_config = config["data"]
    loader_config = config["loader"]
    train_dataset = CocoDetectionDataset(
        data_config["train_images"], data_config["train_annotations"]
    )
    batch_size = _process_batch_size(loader_config["batch_size"], world_size)

    hard_replay = None
    hard_replay_path = (module_config_paths or {}).get("hard_replay")
    if hard_replay_path is not None:
        hard_replay = build_hard_replay_controller_from_yaml(
            hard_replay_path,
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=bool(loader_config["shuffle"]),
            seed=int(config["seed"]),
            rank=rank,
            world_size=world_size,
            arch=arch,
        )

    if hard_replay is not None:
        train_loader = _build_loader(
            train_dataset, loader_config, batch_sampler=hard_replay.batch_sampler
        )
        train_loader.hard_replay = hard_replay
    elif distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(loader_config["shuffle"]),
            seed=int(config["seed"]),
        )
        train_loader = _build_loader(
            train_dataset, loader_config, batch_size=batch_size, sampler=sampler
        )
    else:
        train_loader = _build_loader(
            train_dataset, loader_config, batch_size=batch_size, shuffle=loader_config["shuffle"]
        )

    val_images = data_config.get("val_images")
    val_annotations = data_config.get("val_annotations")
    if not val_images or not val_annotations:
        return train_loader, None
    val_dataset = CocoDetectionDataset(val_images, val_annotations)
    return train_loader, _build_loader(val_dataset, loader_config, batch_size=batch_size)


def build_eval_dataloader(config: dict[str, Any]) -> DataLoader[Any]:
    dataset = CocoDetectionDataset(config["data"]["images"], config["data"]["annotations"])
    return _build_loader(dataset, config["loader"], batch_size=config["loader"]["batch_size"])


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _build_loader(
    dataset: Dataset[Any], loader_config: dict[str, Any], **kwargs: Any
) -> DataLoader[Any]:
    num_workers = loader_config["num_workers"]
    return DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=loader_config["pin_memory"],
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        **kwargs,
    )


def _process_batch_size(global_batch_size: int, world_size: int) -> int:
    batch_size = int(global_batch_size)
    if batch_size < world_size or batch_size % world_size:
        raise ValueError(
            "loader.batch_size is a global batch size and must be a positive multiple of the "
            f"number of training devices; got batch_size={batch_size}, devices={world_size}."
        )
    return batch_size // world_size


def _resolve_image_path(images_dir: Path, file_name: str) -> Path:
    image_path = Path(file_name)
    if image_path.is_absolute():
        return image_path.resolve()

    parts = image_path.parts
    candidates = [image_path]
    for prefix_len in range(min(len(parts), len(images_dir.parts)), 0, -1):
        if parts[:prefix_len] == images_dir.parts[-prefix_len:]:
            if prefix_len < len(parts):
                candidates.append(Path(*parts[prefix_len:]))
            break
    candidates += [Path(*parts[index:]) for index in range(1, len(parts))]

    for candidate in candidates:
        resolved = (images_dir / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (images_dir / image_path).resolve()


def _build_target(
    image_id: int,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    areas: list[float] = []
    crowds: list[int] = []
    annotation_ids: list[int] = []
    for annotation in annotations:
        x, y, w, h = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x1 = max(0.0, float(x))
        y1 = max(0.0, float(y))
        x2 = min(float(width), x1 + max(0.0, float(w)))
        y2 = min(float(height), y1 + max(0.0, float(h)))
        category_id = int(annotation["category_id"])
        if x2 <= x1 or y2 <= y1 or category_id < 1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(category_id)
        areas.append(float(annotation.get("area", (x2 - x1) * (y2 - y1))))
        crowds.append(int(annotation.get("iscrowd", 0)))
        annotation_ids.append(int(annotation["id"]))

    annotation_ids_tensor = torch.tensor(annotation_ids, dtype=torch.int64)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        "annotation_ids": annotation_ids_tensor,
        "gt_ids": annotation_ids_tensor,
    }
