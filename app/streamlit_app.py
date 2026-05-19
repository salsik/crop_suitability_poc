from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px

# Allow imports when running from project root.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from crop_profiles import load_crop_profiles
from utils import class_label, score_to_rgb

st.set_page_config(page_title="Syria Crop Suitability PoC", layout="wide")

DEFAULT_DATA = ROOT / "data" / "processed" / "demo_crop_suitability_predictions.csv"
DEFAULT_DATA = ROOT / "data" / "processed" / "crop_suitability_predictions.csv"
MODEL_DIR = ROOT / "models"


@st.cache_data
def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_profiles() -> dict:
    return load_crop_profiles(ROOT / "config" / "crop_profiles.yaml")


def get_crop_score_column(crop_key: str, use_model: bool) -> str:
    model_col = f"{crop_key}_model_score"
    rule_col = f"{crop_key}_rule_score"
    return model_col if use_model else rule_col


def add_color_column(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    colors = out[score_col].apply(lambda x: score_to_rgb(float(x) if pd.notnull(x) else np.nan))
    out["color"] = colors
    return out


def render_map(df: pd.DataFrame, score_col: str, radius: int = 120) -> None:
    map_df = df.dropna(subset=["longitude", "latitude", score_col]).copy()
    if len(map_df) == 0:
        st.warning("No valid map rows found.")
        return
    map_df = add_color_column(map_df, score_col)

    mid_lat = float(map_df["latitude"].mean())
    mid_lon = float(map_df["longitude"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=radius,
        pickable=True,
        opacity=0.65,
    )

    tooltip = {
        "html": "<b>Score:</b> {" + score_col + "}<br/><b>Best crop:</b> {best_crop}<br/><b>NDVI:</b> {NDVI}<br/><b>NDMI:</b> {NDMI}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=8, pitch=0),
        layers=[layer],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


def show_feature_importance(crop_key: str) -> None:
    path = MODEL_DIR / f"{crop_key}_feature_importance.csv"
    if not path.exists():
        st.info("Feature importance file not found yet. Run training first.")
        return
    imp = pd.read_csv(path).head(12)
    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top feature importance")
    st.plotly_chart(fig, use_container_width=True)


st.title("Syria Crop Suitability PoC")
st.caption("Screen 1: Olive and Damask rose suitability heatmaps from satellite, soil, climate, and terrain features.")

profiles = load_profiles()
crop_keys = list(profiles.keys())

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload prediction CSV", type=["csv"])
    if uploaded is not None:
        df = load_data(uploaded)
        st.success("Loaded uploaded CSV")
    elif DEFAULT_DATA.exists():
        df = load_data(DEFAULT_DATA)
        st.success(f"Loaded default: {DEFAULT_DATA.relative_to(ROOT)}")
    else:
        st.error("No predictions found. Run src/make_demo_data.py and src/train.py first.")
        st.stop()

    st.header("Map settings")
    crop_key = st.selectbox(
        "Crop",
        crop_keys,
        format_func=lambda k: profiles[k].get("display_name", k),
    )
    use_model = st.toggle("Use trained model score", value=True)
    min_score = st.slider("Minimum score", 0, 100, 0)
    radius = st.slider("Point radius", 30, 400, 120, step=10)

score_col = get_crop_score_column(crop_key, use_model)
class_col = score_col.replace("score", "class")

if score_col not in df.columns:
    st.error(f"Column not found: {score_col}. Check prediction pipeline output.")
    st.stop()

filtered = df[df[score_col] >= min_score].copy()

crop_profile = profiles[crop_key]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(filtered):,}")
c2.metric("Mean score", f"{filtered[score_col].mean():.1f}" if len(filtered) else "n/a")
c3.metric("Priority zones", f"{(filtered[score_col] >= 80).sum():,}")
c4.metric("High+ zones", f"{(filtered[score_col] >= 65).sum():,}")

st.subheader(f"{crop_profile.get('display_name', crop_key)} suitability heatmap")
st.write(
    f"**Product:** {crop_profile.get('product', '')}  |  "
    f"**Sun:** {crop_profile.get('sun_requirement', '')}  |  "
    f"**Shade tolerance:** {crop_profile.get('shade_tolerance', '')}"
)
st.info(crop_profile.get("agrivoltaics_note", ""))

render_map(filtered, score_col, radius=radius)

left, right = st.columns([1, 1])
with left:
    st.subheader("Score distribution")
    fig = px.histogram(filtered, x=score_col, nbins=30, title="Suitability score distribution")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Feature importance")
    show_feature_importance(crop_key)

st.subheader("Top candidate locations")
show_cols = [
    "longitude", "latitude", score_col, class_col,
    f"{crop_key}_soil_component", f"{crop_key}_water_component", f"{crop_key}_heat_component",
    f"{crop_key}_vegetation_component", f"{crop_key}_terrain_component",
    "NDVI", "NDMI", "BSI", "soil_ph", "soil_org_carbon", "mean_temp_c", "annual_rain_mm", "slope_deg",
]
show_cols = [c for c in show_cols if c in filtered.columns]
top = filtered.sort_values(score_col, ascending=False).head(50).copy()
if class_col in top.columns:
    top["class_label"] = top[class_col].apply(class_label)
st.dataframe(top[show_cols + (["class_label"] if "class_label" in top.columns else [])], use_container_width=True)

st.subheader("Compare olive vs Damask rose")
compare_cols = ["longitude", "latitude"]
for k in crop_keys:
    for suffix in ["model_score", "model_class", "rule_score", "rule_class"]:
        col = f"{k}_{suffix}"
        if col in df.columns:
            compare_cols.append(col)
for col in ["best_crop", "best_crop_score", "best_crop_class"]:
    if col in df.columns:
        compare_cols.append(col)
st.dataframe(df[compare_cols].sort_values("best_crop_score", ascending=False).head(100), use_container_width=True)

st.caption("This dashboard is a screening tool. Final planting decisions require farmer validation and soil/lab testing.")
