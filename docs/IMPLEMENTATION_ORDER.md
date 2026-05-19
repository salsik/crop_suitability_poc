# Implementation order: Weeks 1 to 7

This document describes the practical order to reach the first olive and Damask rose heatmap.

---

## Week 1: Confirm AOI and crop assumptions

### Tasks

1. Confirm the Homs-Damascus corridor polygon.
2. Add the exact polygon to GEE as an asset.
3. Confirm first two crops:
   - Olive
   - Damask rose
4. Review `config/crop_profiles.yaml` with an agronomist.

### Output

- confirmed AOI
- crop profiles v0.1

---

## Week 2: GEE data extraction

### Tasks

1. Open `gee/01_export_feature_grid.js`.
2. Replace fallback polygon with exact AOI asset.
3. Run the export to Google Drive.
4. Download the CSV and save it as:

```text
data/raw/syria_crop_features_2025.csv
```

### Output

Feature CSV with:

- Sentinel-2 bands
- vegetation/water/moisture/bare-soil indices
- WorldCover label
- soil pH proxy
- soil organic carbon proxy
- slope/elevation
- temperature/rainfall
- latitude/longitude

---

## Week 3: Data quality checks

### Tasks

1. Check missing values.
2. Check coordinate range.
3. Check NDVI/NDMI/BSI distributions.
4. Exclude built-up/water classes.
5. Confirm row count is manageable.

### Command

```bash
python src/train.py --input data/raw/syria_crop_features_2025.csv --output data/processed/crop_suitability_predictions.csv
```

This command includes checks and will fail if required columns are missing.

---

## Week 4: Expert-rule crop suitability score

### Logic

For each crop, the score combines:

- soil suitability
- water/moisture suitability
- heat suitability
- vegetation suitability
- terrain suitability
- land-use eligibility

The result is a 0-100 score and 0-4 class.

### Output columns

```text
olive_rule_score
olive_rule_class
damask_rose_rule_score
damask_rose_rule_class
```

---

## Week 5: Python model training

### Why train if we already have rule scores?

Because at this stage we do not have real crop suitability field labels. The model is trained to reproduce the expert-rule score so we can:

- test the pipeline
- inspect feature importance
- support dashboard demonstration
- later replace pseudo labels with real field labels

### Output

```text
models/olive_suitability_model.joblib
models/damask_rose_suitability_model.joblib
models/olive_feature_importance.csv
models/damask_rose_feature_importance.csv
```

---

## Week 6: Prediction and crop comparison

### Output columns

```text
olive_model_score
olive_model_class
damask_rose_model_score
damask_rose_model_class
best_crop
best_crop_score
best_crop_class
```

### Interpretation

- `rule_score` is the transparent agronomic/expert scoring baseline.
- `model_score` is the machine-learning approximation.
- large disagreement between rule and model should be investigated.

---

## Week 7: Streamlit heatmap dashboard

### Command

```bash
streamlit run app/streamlit_app.py
```

### Dashboard features

- choose crop: olive or Damask rose
- choose model score or rule score
- filter by minimum score
- heatmap points
- score distribution
- feature importance
- top candidate locations
- crop comparison table

---

## After Week 7: real validation

For the farmer's 0.1 hectare plot, collect:

- plot boundary GPS
- soil pH
- organic matter
- EC/salinity
- texture
- irrigation access
- crop history
- photos
- farmer notes

Then create `data/raw/field_validation.csv` and train a real supervised model.
