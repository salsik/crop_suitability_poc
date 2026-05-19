from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from crop_profiles import load_crop_profiles, load_project_config
from suitability import add_all_crop_rule_scores, add_best_crop_columns
from utils import coerce_numeric, ensure_dir, load_feature_columns, read_csv_safely, suitability_class, validate_columns


def train_crop_regressor(
    df: pd.DataFrame,
    crop_key: str,
    feature_cols: list[str],
    cfg: Dict[str, Any],
    model_dir: Path,
) -> tuple[pd.Series, dict]:
    target_col = f"{crop_key}_rule_score"
    X = df[feature_cols]
    y = df[target_col].astype(float)

    training_cfg = cfg.get("training", {})
    test_size = float(training_cfg.get("test_size", 0.25))
    random_state = int(training_cfg.get("random_state", 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=int(training_cfg.get("n_estimators", 250)),
            min_samples_leaf=int(training_cfg.get("min_samples_leaf", 4)),
            max_depth=training_cfg.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test).clip(0, 100)
    pred_all = model.predict(X).clip(0, 100)

    metrics = {
        "crop": crop_key,
        "target": target_col,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "r2": float(r2_score(y_test, pred_test)),
    }

    rf = model.named_steps["rf"]
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    ensure_dir(model_dir)
    joblib.dump(model, model_dir / f"{crop_key}_suitability_model.joblib")
    importance.to_csv(model_dir / f"{crop_key}_feature_importance.csv", index=False)
    with (model_dir / f"{crop_key}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return pd.Series(pred_all, index=df.index, name=f"{crop_key}_model_score"), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop suitability models and export predictions.")
    parser.add_argument("--input", required=True, help="Input CSV exported from GEE or demo generator.")
    parser.add_argument("--output", default="data/processed/crop_suitability_predictions.csv")
    parser.add_argument("--crop-profiles", default="config/crop_profiles.yaml")
    parser.add_argument("--project-config", default="config/project.yaml")
    parser.add_argument("--feature-config", default="config/feature_columns.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--include-ineligible", action="store_true", help="Do not filter built-up/water rows before training.")
    args = parser.parse_args()

    df = read_csv_safely(args.input)
    feature_config = load_feature_columns(args.feature_config)
    required_cols = feature_config["required_base_columns"]
    feature_cols = feature_config["model_feature_columns"]

    validate_columns(df, required_cols)
    numeric_cols = sorted(set(required_cols + feature_cols + ["worldcover_label", "eligible_land"]))
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = coerce_numeric(df, numeric_cols)

    # Drop rows without coordinates or key features.
    df = df.dropna(subset=["longitude", "latitude", "NDVI", "NDMI", "soil_ph", "mean_temp_c"]).copy()

    # Optional filter: keep only candidate agricultural/natural land.
    if not args.include_ineligible:
        if "eligible_land" in df.columns:
            df = df[df["eligible_land"].fillna(0).astype(float) >= 0.5].copy()
        elif "worldcover_label" in df.columns:
            df = df[df["worldcover_label"].isin([10, 20, 30, 40, 60])].copy()

    if len(df) < 100:
        raise ValueError(f"Too few rows after filtering: {len(df)}. Check input CSV or use --include-ineligible.")

    profiles = load_crop_profiles(args.crop_profiles)
    cfg = load_project_config(args.project_config)
    model_dir = Path(args.model_dir)

    scored = add_all_crop_rule_scores(df, profiles)

    all_metrics = []
    for crop_key in profiles.keys():
        pred, metrics = train_crop_regressor(scored, crop_key, feature_cols, cfg, model_dir)
        scored[f"{crop_key}_model_score"] = pred.round(2)
        scored[f"{crop_key}_model_class"] = suitability_class(scored[f"{crop_key}_model_score"])
        all_metrics.append(metrics)

    scored = add_best_crop_columns(scored, list(profiles.keys()))

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    scored.to_csv(output_path, index=False)

    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input": args.input, "output": args.output, "metrics": all_metrics}, f, indent=2)

    print(f"Input rows used: {len(scored):,}")
    print(f"Wrote predictions: {output_path}")
    print("Metrics:")
    for m in all_metrics:
        print(f"  {m['crop']}: MAE={m['mae']:.2f}, R2={m['r2']:.3f}")


if __name__ == "__main__":
    main()
