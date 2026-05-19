#!/usr/bin/env bash
set -e

python src/make_demo_data.py --output data/raw/demo_syria_crop_features.csv --n 5000
python src/train.py --input data/raw/demo_syria_crop_features.csv --output data/processed/demo_crop_suitability_predictions.csv
python src/export_map_layers.py --input data/processed/demo_crop_suitability_predictions.csv --output-dir data/exports

echo "Demo pipeline complete. Run: streamlit run app/streamlit_app.py"
