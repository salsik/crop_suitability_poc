# Data schema

## GEE feature CSV

Expected path:

```text
data/raw/syria_crop_features_2025.csv
```

Required columns:

| Column | Type | Description |
|---|---|---|
| longitude | float | point longitude |
| latitude | float | point latitude |
| BLUE | float | Sentinel-2 B2 reflectance |
| GREEN | float | Sentinel-2 B3 reflectance |
| RED | float | Sentinel-2 B4 reflectance |
| NIR | float | Sentinel-2 B8 reflectance |
| RE1 | float | Sentinel-2 B5 red-edge |
| RE2 | float | Sentinel-2 B6 red-edge |
| RE3 | float | Sentinel-2 B7 red-edge |
| SWIR1 | float | Sentinel-2 B11 |
| SWIR2 | float | Sentinel-2 B12 |
| NDVI | float | vegetation index |
| EVI | float | enhanced vegetation index |
| SAVI | float | soil-adjusted vegetation index |
| NDWI | float | water index |
| NDMI | float | moisture index |
| NDBI | float | built-up index |
| BSI | float | bare soil index |
| soil_ph | float | soil pH proxy |
| soil_org_carbon | float | soil organic carbon proxy |
| slope_deg | float | slope degrees |
| elevation_m | float | SRTM elevation |
| mean_temp_c | float | mean 2m temperature in Celsius |
| annual_rain_mm | float | annual rainfall in mm |

Optional:

| Column | Description |
|---|---|
| worldcover_label | ESA WorldCover class |
| eligible_land | 1 if agricultural/natural candidate land, 0 otherwise |
| system:index | GEE index |
| .geo | GEE geometry JSON |

---

## Prediction CSV

Expected path:

```text
data/processed/crop_suitability_predictions.csv
```

Added columns:

| Column | Description |
|---|---|
| olive_rule_score | expert-rule score 0-100 |
| olive_rule_class | suitability class 0-4 |
| olive_model_score | random-forest score 0-100 |
| olive_model_class | model suitability class 0-4 |
| damask_rose_rule_score | expert-rule score 0-100 |
| damask_rose_rule_class | suitability class 0-4 |
| damask_rose_model_score | random-forest score 0-100 |
| damask_rose_model_class | model suitability class 0-4 |
| best_crop | best crop by model score |
| best_crop_score | highest crop score |
| best_crop_class | class of best score |

Class meanings:

| Class | Meaning |
|---:|---|
| 0 | Very low |
| 1 | Low |
| 2 | Moderate |
| 3 | High |
| 4 | Priority trial zone |
