from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def infer_num_classes_from_runtime_config(config: dict[str, Any]) -> int:
    data = config["data"]
    paths = [
        Path(data[key]).expanduser().resolve()
        for key in ("train_annotations", "val_annotations", "annotations")
        if data.get(key)
    ]
    category_ids: set[int] = set()
    for path in dict.fromkeys(paths):
        category_ids |= _load_category_ids(path)
    return max(category_ids) + 1


def _load_category_ids(path: Path) -> set[int]:
    with open(path, encoding="utf-8") as handle:
        categories = json.load(handle).get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Annotation file must contain a non-empty 'categories' list: {path}")

    category_ids: list[int] = []
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or "id" not in category:
            raise ValueError(
                f"Annotation category entry #{index} must be a mapping with an 'id': {path}"
            )
        category_id = int(category["id"])
        if category_id < 1:
            raise ValueError(
                f"Annotation category IDs must be >= 1 because 0 is reserved for background: {path}"
            )
        category_ids.append(category_id)
    if len(set(category_ids)) != len(category_ids):
        raise ValueError(f"Annotation file contains duplicate category IDs: {path}")
    return set(category_ids)
