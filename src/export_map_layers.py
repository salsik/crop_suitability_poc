from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir, read_csv_safely


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight map layer CSVs for dashboard or sharing.")
    parser.add_argument("--input", default="data/processed/crop_suitability_predictions.csv")
    parser.add_argument("--output-dir", default="data/exports")
    args = parser.parse_args()

    df = read_csv_safely(args.input)
    out_dir = ensure_dir(args.output_dir)

    crops = [c.replace("_model_score", "") for c in df.columns if c.endswith("_model_score")]
    keep_base = ["longitude", "latitude", "worldcover_label", "eligible_land"]
    keep_base = [c for c in keep_base if c in df.columns]

    for crop in crops:
        cols = keep_base + [
            f"{crop}_model_score", f"{crop}_model_class", f"{crop}_rule_score",
            f"{crop}_soil_component", f"{crop}_water_component", f"{crop}_heat_component",
            f"{crop}_vegetation_component", f"{crop}_terrain_component"
        ]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(out_dir / f"{crop}_heatmap_points.csv", index=False)

    print(f"Exported crop map layers to: {out_dir}")


if __name__ == "__main__":
    main()
