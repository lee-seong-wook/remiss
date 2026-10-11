from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SplitConfig:
    name: str
    images_dir: Path
    labels_dir: Path
    output_json: Path


@dataclass
class SplitStats:
    name: str
    image_count: int = 0
    annotation_count: int = 0
    missing_label_files: int = 0
    empty_label_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO detection dataset into COCO annotations."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the YOLO dataset root. Expected to contain a dataset YAML plus images/ and labels/ directories.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset YAML path. Defaults to <data>/<dataset-name>.yaml, data.yaml, or the single YAML file in the root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where COCO JSON files will be written. Defaults to <data>/annotations.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Dataset splits to convert. Defaults to every split present in the dataset YAML among train/val/test.",
    )
    return parser.parse_args()


def resolve_dataset_yaml(dataset_root: Path, explicit_config: str | None) -> Path:
    if explicit_config is not None:
        config_path = Path(explicit_config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {config_path}")
        return config_path

    candidates = [
        dataset_root / f"{dataset_root.name}.yaml",
        dataset_root / f"{dataset_root.name}.yml",
        dataset_root / "data.yaml",
        dataset_root / "data.yml",
        dataset_root / "dataset.yaml",
        dataset_root / "dataset.yml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    yaml_files = sorted(dataset_root.glob("*.yaml")) + sorted(dataset_root.glob("*.yml"))
    if len(yaml_files) == 1:
        return yaml_files[0].resolve()
    if not yaml_files:
        raise FileNotFoundError(f"No dataset YAML found under {dataset_root}")
    raise FileNotFoundError(
        "Multiple dataset YAML files found. Pass --config explicitly: "
        + ", ".join(str(path) for path in yaml_files)
    )


def load_dataset_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Dataset YAML must contain a mapping at the top level: {config_path}")
    return data


def build_categories(names: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    if isinstance(names, Mapping):
        items = sorted(
            ((int(key), str(value)) for key, value in names.items()), key=lambda item: item[0]
        )
    elif isinstance(names, list):
        items = [(index, str(value)) for index, value in enumerate(names)]
    else:
        raise TypeError("Dataset YAML must define 'names' as a mapping or a list")

    categories: list[dict[str, Any]] = []
    class_to_category_id: dict[int, int] = {}
    for class_index, class_name in items:
        category_id = class_index + 1
        class_to_category_id[class_index] = category_id
        categories.append({"id": category_id, "name": class_name, "supercategory": "object"})
    return categories, class_to_category_id


def resolve_split_configs(
    dataset_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    requested_splits: list[str] | None,
) -> list[SplitConfig]:
    split_names = requested_splits or [
        name for name in ("train", "val", "test") if config.get(name) is not None
    ]
    if not split_names:
        raise ValueError("No dataset splits found in the dataset YAML")

    split_configs: list[SplitConfig] = []
    for split_name in split_names:
        split_value = config.get(split_name)
        if split_value is None:
            raise ValueError(f"Split '{split_name}' is not defined in the dataset YAML")
        if not isinstance(split_value, str):
            raise TypeError(f"Split '{split_name}' must resolve to an image directory path")

        images_dir = resolve_path(dataset_root, split_value)
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory for split '{split_name}' not found: {images_dir}"
            )

        labels_dir = resolve_labels_dir(dataset_root, images_dir)
        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory for split '{split_name}' not found: {labels_dir}"
            )

        split_configs.append(
            SplitConfig(
                name=split_name,
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_json=output_dir / f"instances_{split_name}.json",
            )
        )
    return split_configs


def resolve_path(dataset_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (dataset_root / candidate).resolve()


def resolve_labels_dir(dataset_root: Path, images_dir: Path) -> Path:
    try:
        relative_images_dir = images_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Image directory must live under the dataset root: {images_dir}") from exc

    if not relative_images_dir.parts or relative_images_dir.parts[0] != "images":
        raise ValueError(
            "This converter expects split image directories under '<data>/images/...'. "
            f"Received: {images_dir}"
        )

    return (dataset_root / "labels" / Path(*relative_images_dir.parts[1:])).resolve()


def list_images(images_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")
    return images


def label_path_for_image(image_path: Path, *, images_dir: Path, labels_dir: Path) -> Path:
    return (labels_dir / image_path.relative_to(images_dir)).with_suffix(".txt")


def convert_split(
    split: SplitConfig,
    categories: list[dict[str, Any]],
    class_to_category_id: Mapping[int, int],
    *,
    dataset_name: str,
) -> SplitStats:
    images = list_images(split.images_dir)
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    stats = SplitStats(name=split.name)
    annotation_id = 1

    for image_id, image_path in enumerate(images, start=1):
        width, height = read_image_size(image_path)
        file_name = image_path.relative_to(split.images_dir).as_posix()
        coco_images.append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
            }
        )
        stats.image_count += 1

        label_path = label_path_for_image(
            image_path, images_dir=split.images_dir, labels_dir=split.labels_dir
        )
        if not label_path.exists():
            stats.missing_label_files += 1
            continue

        label_text = label_path.read_text(encoding="utf-8").strip()
        if not label_text:
            stats.empty_label_files += 1
            continue

        for line_number, line in enumerate(label_text.splitlines(), start=1):
            annotation = parse_yolo_annotation(
                line=line,
                line_number=line_number,
                label_path=label_path,
                image_id=image_id,
                annotation_id=annotation_id,
                image_width=width,
                image_height=height,
                class_to_category_id=class_to_category_id,
            )
            if annotation is None:
                continue
            coco_annotations.append(annotation)
            stats.annotation_count += 1
            annotation_id += 1

    split.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {
            "description": f"{dataset_name} {split.name} split converted from YOLO to COCO",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    with split.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(
        f"[{split.name}] wrote {split.output_json} "
        f"(images={stats.image_count}, annotations={stats.annotation_count}, "
        f"missing_labels={stats.missing_label_files}, empty_labels={stats.empty_label_files})"
    )
    print(f"[{split.name}] images root for COCO loading: {split.images_dir}")
    print(f"[{split.name}] labels root: {split.labels_dir}")
    return stats


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def parse_yolo_annotation(
    *,
    line: str,
    line_number: int,
    label_path: Path,
    image_id: int,
    annotation_id: int,
    image_width: int,
    image_height: int,
    class_to_category_id: Mapping[int, int],
) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) < 5:
        raise ValueError(
            f"Invalid YOLO annotation at {label_path}:{line_number}: expected at least 5 fields"
        )

    class_index = int(parts[0])
    category_id = class_to_category_id.get(class_index)
    if category_id is None:
        raise ValueError(f"Unknown class index {class_index} at {label_path}:{line_number}")

    center_x = float(parts[1]) * image_width
    center_y = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x_min = max(0.0, center_x - (box_width / 2.0))
    y_min = max(0.0, center_y - (box_height / 2.0))
    x_max = min(float(image_width), center_x + (box_width / 2.0))
    y_max = min(float(image_height), center_y + (box_height / 2.0))

    clipped_width = max(0.0, x_max - x_min)
    clipped_height = max(0.0, y_max - y_min)
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None

    bbox = [round(x_min, 4), round(y_min, 4), round(clipped_width, 4), round(clipped_height, 4)]
    area = round(clipped_width * clipped_height, 4)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.data).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config_path = resolve_dataset_yaml(dataset_root, args.config)
    dataset_config = load_dataset_config(config_path)
    categories, class_to_category_id = build_categories(dataset_config.get("names"))

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (dataset_root / "annotations")
    )
    requested_splits = [split.strip() for split in args.splits] if args.splits else None
    split_configs = resolve_split_configs(
        dataset_root, dataset_config, output_dir, requested_splits
    )

    print(f"dataset_root: {dataset_root}")
    print(f"config: {config_path}")
    print(f"output_dir: {output_dir}")
    print(f"categories: {len(categories)}")
    print(
        "note: generated COCO category IDs start at 1. "
        f"If you set model.num_classes manually for this dataset, use {len(categories) + 1} to include background."
    )

    for split in split_configs:
        convert_split(
            split,
            categories,
            class_to_category_id,
            dataset_name=dataset_root.name,
        )


if __name__ == "__main__":
    main()
