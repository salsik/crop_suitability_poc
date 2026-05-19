from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils import suitability_class


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def trapezoid(x: pd.Series, min_val: float, opt_low: float, opt_high: float, max_val: float) -> pd.Series:
    """0-1 trapezoid suitability curve."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    rising = (x - min_val) / max(opt_low - min_val, 1e-9)
    falling = (max_val - x) / max(max_val - opt_high, 1e-9)
    score = np.minimum(np.minimum(rising, 1.0), falling)
    return pd.Series(np.clip(score, 0, 1), index=x.index)


def high_good(x: pd.Series, low: float, high: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return pd.Series(np.clip((x - low) / max(high - low, 1e-9), 0, 1), index=x.index)


def low_good(x: pd.Series, low: float, high: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return pd.Series(np.clip((high - x) / max(high - low, 1e-9), 0, 1), index=x.index)


def eligible_land_score(df: pd.DataFrame) -> pd.Series:
    """Return 1 for candidate agricultural/natural land, 0 for built-up/water/etc."""
    if "eligible_land" in df.columns:
        return pd.to_numeric(df["eligible_land"], errors="coerce").fillna(0).clip(0, 1)
    if "worldcover_label" in df.columns:
        wc = pd.to_numeric(df["worldcover_label"], errors="coerce")
        eligible = wc.isin([10, 20, 30, 40, 60]).astype(float)
        return pd.Series(eligible, index=df.index)
    return pd.Series(1.0, index=df.index)


def compute_crop_rule_score(df: pd.DataFrame, profile: Dict[str, Any], prefix: str) -> pd.DataFrame:
    """Compute expert-rule crop suitability score and components.

    This is first-screening logic, not a field-validated agronomic model.
    """
    out = df.copy()
    thresholds = profile["thresholds"]
    weights = profile["weights"]

    soil_ph = _safe_series(out, "soil_ph", 7.2)
    soc = _safe_series(out, "soil_org_carbon", 12.0)
    ndmi = _safe_series(out, "NDMI", 0.0)
    ndvi = _safe_series(out, "NDVI", 0.3)
    temp = _safe_series(out, "mean_temp_c", 24.0)
    rain = _safe_series(out, "annual_rain_mm", 400.0)
    slope = _safe_series(out, "slope_deg", 5.0)
    land = eligible_land_score(out)

    ph_suit = trapezoid(soil_ph, *thresholds["ph"])
    soc_suit = high_good(soc, *thresholds["soil_org_carbon"])
    soil_suit = (ph_suit * 0.60 + soc_suit * 0.40).clip(0, 1)

    moisture_suit = trapezoid(ndmi, *thresholds["ndmi"])
    rain_suit = trapezoid(rain, *thresholds["annual_rain_mm"])
    water_suit = (moisture_suit * 0.65 + rain_suit * 0.35).clip(0, 1)

    heat_suit = trapezoid(temp, *thresholds["mean_temp_c"])
    veg_suit = trapezoid(ndvi, *thresholds["ndvi"])
    terrain_suit = low_good(slope, 0, thresholds["slope_deg_max"])

    score = (
        soil_suit * weights["soil"] +
        water_suit * weights["moisture"] +
        heat_suit * weights["heat"] +
        veg_suit * weights["vegetation"] +
        terrain_suit * weights["terrain"] +
        land * weights["land_use"]
    ) * 100

    # Penalize non-eligible land hard but keep rows for dashboard context.
    score = np.where(land >= 0.5, score, np.minimum(score, 25))

    out[f"{prefix}_soil_component"] = (soil_suit * 100).round(2)
    out[f"{prefix}_water_component"] = (water_suit * 100).round(2)
    out[f"{prefix}_heat_component"] = (heat_suit * 100).round(2)
    out[f"{prefix}_vegetation_component"] = (veg_suit * 100).round(2)
    out[f"{prefix}_terrain_component"] = (terrain_suit * 100).round(2)
    out[f"{prefix}_land_use_component"] = (land * 100).round(2)
    out[f"{prefix}_rule_score"] = pd.Series(score, index=out.index).clip(0, 100).round(2)
    out[f"{prefix}_rule_class"] = suitability_class(out[f"{prefix}_rule_score"])
    return out


def add_all_crop_rule_scores(df: pd.DataFrame, profiles: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for crop_key, profile in profiles.items():
        out = compute_crop_rule_score(out, profile, crop_key)
    return out


def add_best_crop_columns(df: pd.DataFrame, crop_keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    score_cols = [f"{c}_model_score" if f"{c}_model_score" in out.columns else f"{c}_rule_score" for c in crop_keys]
    score_matrix = out[score_cols].to_numpy(dtype=float)
    best_idx = np.nanargmax(score_matrix, axis=1)
    best_scores = np.nanmax(score_matrix, axis=1)
    out["best_crop"] = [crop_keys[i] for i in best_idx]
    out["best_crop_score"] = np.round(best_scores, 2)
    out["best_crop_class"] = suitability_class(out["best_crop_score"])
    return out
