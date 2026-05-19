from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_feature_columns(path: str | Path = "config/feature_columns.json") -> dict:
    p = project_path(path) if not Path(path).is_absolute() else Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nCheck GEE export or config/feature_columns.json."
        )
    return list(required)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def suitability_class(score: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    return np.select(
        [arr >= 80, arr >= 65, arr >= 50, arr >= 35],
        [4, 3, 2, 1],
        default=0,
    ).astype(int)


def class_label(value: int) -> str:
    labels = {
        0: "Very low",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Priority trial zone",
    }
    return labels.get(int(value), "Unknown")


def score_to_rgb(score: float) -> list[int]:
    """Return simple green/yellow/red color for pydeck."""
    if np.isnan(score):
        return [160, 160, 160]
    if score >= 80:
        return [0, 100, 0]
    if score >= 65:
        return [127, 255, 0]
    if score >= 50:
        return [255, 215, 0]
    if score >= 35:
        return [255, 140, 0]
    return [139, 0, 0]
