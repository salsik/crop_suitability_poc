from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dir


def make_demo_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Iizuka, Sosa, Chiba Prefecture bounding box.
    lon = rng.uniform(140.535, 140.569, n)
    lat = rng.uniform(35.718, 35.746, n)

    # Create synthetic gradients that mimic local agricultural variability.
    northness = (lat - lat.min()) / max(lat.max() - lat.min(), 1e-9)
    eastness = (lon - lon.min()) / max(lon.max() - lon.min(), 1e-9)

    # Spectral indices representing fields, forests, solar panels, and water
    ndvi = np.clip(0.35 + 0.40 * northness - 0.15 * eastness + rng.normal(0, 0.08, n), -0.1, 0.85)
    ndmi = np.clip(0.05 + 0.25 * northness - 0.08 * eastness + rng.normal(0, 0.08, n), -0.30, 0.55)
    ndwi = np.clip(-0.10 + 0.15 * northness + rng.normal(0, 0.07, n), -0.5, 0.4)
    bsi = np.clip(0.20 - 0.20 * ndvi + rng.normal(0, 0.08, n), -0.2, 0.7)
    savi = np.clip(ndvi * 0.9 + rng.normal(0, 0.03, n), -0.1, 0.8)
    evi = np.clip(ndvi * 0.85 + rng.normal(0, 0.04, n), -0.1, 0.8)
    ndbi = np.clip(0.05 + 0.20 * bsi - 0.15 * ndvi + rng.normal(0, 0.05, n), -0.5, 0.6)

    # Optical bands scaled like Sentinel-2 (0-1 range)
    red = np.clip(0.08 + 0.10 * bsi - 0.04 * ndvi + rng.normal(0, 0.02, n), 0.01, 0.45)
    nir = np.clip(red * (1 + ndvi) / np.maximum(1 - ndvi, 0.05), 0.01, 0.70)
    green = np.clip(red * 0.95 + rng.normal(0, 0.015, n), 0.01, 0.40)
    blue = np.clip(red * 0.85 + rng.normal(0, 0.015, n), 0.01, 0.35)
    re1 = np.clip((red + nir) / 2 + rng.normal(0, 0.02, n), 0.01, 0.60)
    re2 = np.clip(re1 + 0.03 + rng.normal(0, 0.02, n), 0.01, 0.65)
    re3 = np.clip(re2 + 0.03 + rng.normal(0, 0.02, n), 0.01, 0.70)
    swir1 = np.clip(nir * (1 - ndmi) / np.maximum(1 + ndmi, 0.05), 0.01, 0.70)
    swir2 = np.clip(swir1 + rng.normal(0, 0.04, n), 0.01, 0.75)

    # Sosa, Chiba climate and soil characteristics
    soil_ph = np.clip(6.2 + 0.3 * eastness + rng.normal(0, 0.25, n), 5.0, 7.8)
    soc = np.clip(22 + 8 * northness - 3 * eastness + rng.normal(0, 4, n), 5, 45)
    slope = np.clip(rng.gamma(1.5, 2.5, n), 0, 15)  # relatively flat area
    elev = np.clip(35 + 25 * northness + rng.normal(0, 10, n), 5, 95)  # flat/gentle plain elevation
    temp = np.clip(15.5 + 0.5 * northness + rng.normal(0, 0.4, n), 14.0, 17.0)  # ~15.5C annual mean
    rain = np.clip(1560 + 50 * northness + rng.normal(0, 60, n), 1300, 1850)  # ~1560mm annual rain

    # ESA WorldCover classes: 10 (Tree), 20 (Shrub), 30 (Grass), 40 (Crop), 50 (Built-up), 60 (Bare), 80 (Water)
    # Higher percentage of agricultural crop (40) and tree cover (10) for rural Chiba
    labels = rng.choice([10, 20, 30, 40, 50, 60, 80], n, p=[0.35, 0.05, 0.08, 0.42, 0.06, 0.02, 0.02])
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
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for Japan (Sosa Iizuka) crop suitability PoC.")
    parser.add_argument("--output", default="data/raw/demo_sosa_crop_features.csv")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_demo_data(n=args.n, seed=args.seed)
    out = Path(args.output)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(f"Wrote demo data: {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
