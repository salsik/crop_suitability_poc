from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from crop_profiles import load_crop_profiles
from suitability import add_all_crop_rule_scores, add_best_crop_columns
from utils import coerce_numeric, ensure_dir, load_feature_columns, read_csv_safely, suitability_class, validate_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crop suitability using trained models.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/crop_suitability_predictions_new.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--crop-profiles", default="config/crop_profiles.yaml")
    parser.add_argument("--feature-config", default="config/feature_columns.json")
    args = parser.parse_args()

    df = read_csv_safely(args.input)
    feature_config = load_feature_columns(args.feature_config)
    required_cols = feature_config["required_base_columns"]
    feature_cols = feature_config["model_feature_columns"]
    validate_columns(df, required_cols)

    numeric_cols = sorted(set(required_cols + feature_cols + ["worldcover_label", "eligible_land"]))
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = coerce_numeric(df, numeric_cols).dropna(subset=["longitude", "latitude"]).copy()

    profiles = load_crop_profiles(args.crop_profiles)
    scored = add_all_crop_rule_scores(df, profiles)

    model_dir = Path(args.model_dir)
    for crop_key in profiles.keys():
        model_path = model_dir / f"{crop_key}_suitability_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run src/train.py first.")
        model = joblib.load(model_path)
        scored[f"{crop_key}_model_score"] = model.predict(scored[feature_cols]).clip(0, 100).round(2)
        scored[f"{crop_key}_model_class"] = suitability_class(scored[f"{crop_key}_model_score"])

    scored = add_best_crop_columns(scored, list(profiles.keys()))
    out = Path(args.output)
    ensure_dir(out.parent)
    scored.to_csv(out, index=False)
    print(f"Wrote predictions: {out} ({len(scored):,} rows)")


if __name__ == "__main__":
    main()
