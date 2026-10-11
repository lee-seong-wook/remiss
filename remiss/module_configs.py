from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODULE_CONFIG_PATHS = {
    "missbank": PROJECT_ROOT / "configs" / "remiss" / "missbank.yaml",
    "hard_replay": PROJECT_ROOT / "configs" / "remiss" / "hard_replay_r0050.yaml",
}


def resolve_module_config_paths(
    overrides: Mapping[str, str | Path | None] | None = None,
    *,
    require_exists: bool = True,
) -> dict[str, Path]:
    paths = dict(DEFAULT_MODULE_CONFIG_PATHS)
    for name, raw_path in (overrides or {}).items():
        if name not in paths:
            raise KeyError(f"Unsupported module config key: {name!r}")
        if raw_path is not None:
            paths[name] = Path(raw_path).expanduser().resolve()

    if require_exists:
        missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Module config file was not found: " + ", ".join(missing))
    return paths


def serialize_module_config_paths(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {name: str(Path(path).expanduser().resolve()) for name, path in paths.items()}
