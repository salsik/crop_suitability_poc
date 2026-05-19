#!/usr/bin/env bash
set -e

INPUT=${1:-data/raw/syria_crop_features_2025.csv}
OUTPUT=${2:-data/processed/crop_suitability_predictions.csv}

python src/train.py --input "$INPUT" --output "$OUTPUT"
python src/export_map_layers.py --input "$OUTPUT" --output-dir data/exports

echo "Training pipeline complete. Run: streamlit run app/streamlit_app.py"
