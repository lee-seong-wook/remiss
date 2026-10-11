from __future__ import annotations

ARCH_ALIASES = {
    "faster_rcnn": "fasterrcnn",
    "faster-rcnn": "fasterrcnn",
    "fasterrcnn": "fasterrcnn",
    "fcos": "fcos",
    "dino": "dino",
}


def normalize_arch(raw_arch: str | None) -> str | None:
    if raw_arch is None:
        return None
    value = str(raw_arch).lower()
    return ARCH_ALIASES.get(value, value)
