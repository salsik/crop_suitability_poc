from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dir


def make_demo_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Rough Homs-Damascus corridor bounding box.
    lon = rng.uniform(36.05, 37.05, n)
    lat = rng.uniform(33.25, 34.85, n)

    # Create synthetic gradients that roughly mimic dryland variability.
    northness = (lat - lat.min()) / (lat.max() - lat.min())
    eastness = (lon - lon.min()) / (lon.max() - lon.min())

    ndvi = np.clip(0.18 + 0.35 * northness - 0.15 * eastness + rng.normal(0, 0.08, n), -0.1, 0.85)
    ndmi = np.clip(-0.12 + 0.30 * northness - 0.10 * eastness + rng.normal(0, 0.08, n), -0.45, 0.55)
    ndwi = np.clip(-0.20 + 0.25 * northness + rng.normal(0, 0.07, n), -0.6, 0.4)
    bsi = np.clip(0.28 - 0.25 * ndvi + rng.normal(0, 0.08, n), -0.2, 0.7)
    savi = np.clip(ndvi * 0.9 + rng.normal(0, 0.03, n), -0.1, 0.8)
    evi = np.clip(ndvi * 0.85 + rng.normal(0, 0.04, n), -0.1, 0.8)
    ndbi = np.clip(0.10 + 0.25 * bsi - 0.20 * ndvi + rng.normal(0, 0.05, n), -0.5, 0.6)

    red = np.clip(0.12 + 0.15 * bsi - 0.05 * ndvi + rng.normal(0, 0.02, n), 0.02, 0.45)
    nir = np.clip(red * (1 + ndvi) / np.maximum(1 - ndvi, 0.05), 0.02, 0.70)
    green = np.clip(red * 0.9 + rng.normal(0, 0.015, n), 0.02, 0.40)
    blue = np.clip(red * 0.8 + rng.normal(0, 0.015, n), 0.02, 0.35)
    re1 = np.clip((red + nir) / 2 + rng.normal(0, 0.02, n), 0.02, 0.60)
    re2 = np.clip(re1 + 0.03 + rng.normal(0, 0.02, n), 0.02, 0.65)
    re3 = np.clip(re2 + 0.03 + rng.normal(0, 0.02, n), 0.02, 0.70)
    swir1 = np.clip(nir * (1 - ndmi) / np.maximum(1 + ndmi, 0.05), 0.02, 0.70)
    swir2 = np.clip(swir1 + rng.normal(0, 0.04, n), 0.02, 0.75)

    soil_ph = np.clip(7.2 + 0.4 * eastness + rng.normal(0, 0.25, n), 5.7, 8.8)
    soc = np.clip(12 + 10 * northness - 4 * eastness + rng.normal(0, 4, n), 1, 45)
    slope = np.clip(rng.gamma(2.0, 3.0, n), 0, 30)
    elev = np.clip(500 + 600 * northness + 200 * eastness + rng.normal(0, 80, n), 300, 1500)
    temp = np.clip(25 - 3 * northness + 2 * eastness + rng.normal(0, 1.5, n), 12, 38)
    rain = np.clip(260 + 300 * northness - 80 * eastness + rng.normal(0, 60, n), 80, 900)

    # WorldCover-like labels. Most points are eligible land.
    labels = rng.choice([10, 20, 30, 40, 50, 60, 80], n, p=[0.18, 0.12, 0.12, 0.38, 0.08, 0.10, 0.02])
    eligible = np.isin(labels, [10, 20, 30, 40, 60]).astype(int)

    return pd.DataFrame({
        "system:index": np.arange(n),
        "longitude": lon,
        "latitude": lat,
        "BLUE": blue,
        "GREEN": green,
        "RED": red,
        "NIR": nir,
        "RE1": re1,
        "RE2": re2,
        "RE3": re3,
        "SWIR1": swir1,
        "SWIR2": swir2,
        "NDVI": ndvi,
        "EVI": evi,
        "SAVI": savi,
        "NDWI": ndwi,
        "NDMI": ndmi,
        "NDBI": ndbi,
        "BSI": bsi,
        "soil_ph": soil_ph,
        "soil_org_carbon": soc,
        "slope_deg": slope,
        "elevation_m": elev,
        "mean_temp_c": temp,
        "annual_rain_mm": rain,
        "worldcover_label": labels,
        "eligible_land": eligible,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for the crop suitability PoC.")
    parser.add_argument("--output", default="data/raw/demo_syria_crop_features.csv")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_demo_data(n=args.n, seed=args.seed)
    out = Path(args.output)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(f"Wrote demo data: {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
