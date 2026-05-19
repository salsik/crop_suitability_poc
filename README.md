# Syria Crop Suitability PoC

**Goal:** build a first-screening crop suitability heatmap for **olive** and **Damask rose** in the Homs-Damascus corridor using:

1. **Google Earth Engine (GEE)** for data extraction
2. **Python** for crop scoring, model training, prediction, and exports
3. **Streamlit** for a dashboard heatmap

This project follows the same practical structure as the UNICEF climate-health PoC: satellite feature extraction, open labels/proxy data, Python model pipeline, and an interactive dashboard.

---

## What this PoC does

The first version answers:

> Where are the most suitable zones for olive and Damask rose, based on satellite, terrain, climate, and soil proxy features?

It produces:

- `data/raw/syria_crop_features_2025.csv` from GEE
- `data/processed/crop_suitability_predictions.csv`
- trained pseudo-supervised models in `models/`
- Streamlit dashboard heatmaps for olive and Damask rose

---

## Important scientific note

This PoC does **not** claim to replace agronomists, soil testing, or farmer validation.

The first model uses an **expert-rule crop suitability score** to create first labels, then trains a Python model to reproduce and generalize that score. This is useful for a PoC and dashboard, but it is still **pseudo-supervised** until replaced or calibrated with real field labels such as:

- soil pH lab test
- soil organic matter
- EC/salinity
- texture
- farmer yield observations
- irrigation records
- crop survival/growth observations

Once field labels exist, the same pipeline can train a real supervised model.

---

## Project structure

```text
syria_crop_suitability_poc/
  README.md
  requirements.txt
  .env.example
  config/
    crop_profiles.yaml
    project.yaml
    feature_columns.json
  gee/
    01_export_feature_grid.js
    02_export_worldcover_training_points.js
    03_export_farmer_plot_features.js
  data/
    raw/
    processed/
    exports/
  src/
    crop_profiles.py
    suitability.py
    train.py
    predict.py
    make_demo_data.py
    export_map_layers.py
    utils.py
  notebooks/
    01_training_crop_suitability.ipynb
  app/
    streamlit_app.py
  docs/
    IMPLEMENTATION_ORDER.md
    DATA_SCHEMA.md
    FIELD_VALIDATION_TEMPLATE.md
  scripts/
    run_demo_pipeline.sh
    run_training_pipeline.sh
```

---

## Quick demo without GEE

Use this to test the full Python + Streamlit flow with synthetic demo data:

```bash
pip install -r requirements.txt
python src/make_demo_data.py --output data/raw/demo_syria_crop_features.csv --n 5000
python src/train.py --input data/raw/demo_syria_crop_features.csv --output data/processed/crop_suitability_predictions.csv
streamlit run app/streamlit_app.py
```

---

## Real workflow with GEE

1. Open `gee/01_export_feature_grid.js` in Google Earth Engine.
2. Replace the AOI with your exact corridor boundary asset if available.
3. Run the export task to Google Drive.
4. Download the exported CSV and save it as:

```text
data/raw/syria_crop_features_2025.csv
```

5. Run:

```bash
python src/train.py --input data/raw/syria_crop_features_2025.csv --output data/processed/crop_suitability_predictions.csv
streamlit run app/streamlit_app.py
```

---

## Main input features

The GEE export includes the Sentinel-2 features used in the previous PoC style:

```text
BLUE, GREEN, RED, NIR, RE1, RE2, RE3, SWIR1, SWIR2,
NDVI, EVI, SAVI, NDWI, NDMI, NDBI, BSI
```

This project adds practical crop suitability features:

```text
soil_ph, soil_org_carbon, slope_deg, elevation_m,
mean_temp_c, annual_rain_mm, worldcover_label, latitude, longitude
```

---

## Output fields

The training pipeline produces:

```text
olive_rule_score
olive_rule_class
olive_model_score
olive_model_class
damask_rose_rule_score
damask_rose_rule_class
damask_rose_model_score
damask_rose_model_class
best_crop
best_crop_score
```

Suitability classes:

| Class | Meaning |
|---:|---|
| 0 | Very low |
| 1 | Low |
| 2 | Moderate |
| 3 | High |
| 4 | Priority trial zone |

---

## Recommended next validation step

For the 0.1 ha farmer trial plot, collect:

- GPS polygon of the land
- soil pH
- organic matter
- EC/salinity
- soil texture
- current crop history
- irrigation source and frequency
- farmer notes
- photos

Then update `data/raw/field_validation.csv` and use the template in `docs/FIELD_VALIDATION_TEMPLATE.md`.
