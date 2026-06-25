# Step 00 - Configuration
#
# Role:
#   Load project settings from configs/config.yaml.
#
# Why this exists:
#   Other pipeline files should not hardcode paths, model names, chunk sizes,
#   or retrieval settings. They should read those values from one config file.
#
# Input:
#   configs/config.yaml
#
# Output:
#   A Python dictionary with settings, plus helper functions for resolving
#   project-relative paths like "data/raw_pdfs" into absolute paths.

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config file and return it as a Python dictionary."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config


def project_path(path_value: str | Path) -> Path:
    """Convert a project-relative path from config.yaml into an absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve only the values inside the config 'paths' section."""
    return {
        name: project_path(path_value)
        for name, path_value in config.get("paths", {}).items()
    }


def main() -> None:
    """Small sanity check we can run before building the rest of the project."""
    config = load_config()
    resolved_paths = resolve_config_paths(config)

    print("Project root:")
    print(PROJECT_ROOT)

    print("\nConfig:")
    pprint(config)

    print("\nResolved paths:")
    pprint(resolved_paths)


if __name__ == "__main__":
    main()
