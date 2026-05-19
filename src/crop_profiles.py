from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_crop_profiles(path: str | Path = "config/crop_profiles.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Crop profile file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Crop profile YAML is empty or invalid.")
    return profiles


def load_project_config(path: str | Path = "config/project.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
