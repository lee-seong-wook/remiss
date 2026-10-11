from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .arch import normalize_arch
from .config import load_yaml_file
from .hard_replay import load_hard_replay_config
from .missbank import load_missbank_config
from .module_configs import resolve_module_config_paths

CONFIG_LOADERS = {
    "missbank": load_missbank_config,
    "hard_replay": load_hard_replay_config,
}


def collect_enabled_module_configs(
    arch: str | None,
    *,
    config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    normalized_arch = normalize_arch(arch)
    paths = resolve_module_config_paths(config_paths, require_exists=False)
    snapshots: dict[str, Any] = {}
    for name, load_config in CONFIG_LOADERS.items():
        path = Path(paths[name]).expanduser().resolve()
        if not path.is_file():
            continue
        if not load_config(path, arch=normalized_arch).enabled:
            continue
        snapshots[name] = load_yaml_file(path)
    return snapshots
