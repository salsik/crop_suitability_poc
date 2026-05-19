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


def crop_display_name(crop_key: str, profiles: dict) -> str:
    return profiles[crop_key].get("display_name", crop_key)


def get_crop_score_column(crop_key: str, use_model: bool) -> str:
    model_col = f"{crop_key}_model_score"
    rule_col = f"{crop_key}_rule_score"
    return model_col if use_model else rule_col


def add_color_column(
    df: pd.DataFrame,
    score_col: str,
    color_col: str = "color",
) -> pd.DataFrame:
    out = df.copy()
    colors = out[score_col].apply(
        lambda x: score_to_rgb(float(x) if pd.notnull(x) else np.nan)
    )
    out[color_col] = colors
    return out


def score_legend_html_1onldi() -> str:
    """Legend uses the same score_to_rgb() helper used by the map."""
    bins = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    rows = []
    for label, value_range, representative_score in bins:
        r, g, b = score_to_rgb(representative_score)
        rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin:0.20rem 0;">
                <span style="width:16px;height:16px;border-radius:3px;background:rgb({r},{g},{b});
                             border:1px solid rgba(0,0,0,0.25);display:inline-block;"></span>
                <span><b>{label}</b> <span style="color:#666;">({value_range})</span></span>
            </div>
            """
        )

    return (
        """
        <div style="padding:0.75rem 0.9rem;border:1px solid rgba(49,51,63,0.2);
                    border-radius:0.5rem;margin:0.25rem 0 1rem 0;background:rgba(250,250,250,0.65);">
            <div style="font-weight:700;margin-bottom:0.35rem;">Map color legend</div>
        """
        + "\n".join(rows)
        + """
            <div style="font-size:0.85rem;color:#666;margin-top:0.45rem;">
                Colors represent suitability score. When multiple crops are selected, the color uses the
                best score among the selected crops at that location.
            </div>
        </div>
        """
    )


def render_score_legend_1on() -> None:
    st.markdown("**Map color legend**")

    legend_items = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    for label, value_range, representative_score in legend_items:
        r, g, b = score_to_rgb(representative_score)

        cols = st.columns([0.08, 0.92])
        with cols[0]:
            st.color_picker(
                label=f"{label} color",
                value=f"#{r:02x}{g:02x}{b:02x}",
                label_visibility="collapsed",
                disabled=True,
                key=f"legend_{label}",
            )

        with cols[1]:
            st.markdown(f"**{label}** ({value_range})")

    st.caption(
        "Colors represent suitability score. When multiple crops are selected, "
        "the color uses the best score among the selected crops at that location."
    )

def render_score_legend() -> None:
    st.markdown("**Map color legend**")

    legend_items = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    for label, value_range, representative_score in legend_items:
        r, g, b = score_to_rgb(representative_score)

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">
                <span style="
                    width:16px;
                    height:16px;
                    border-radius:3px;
                    background-color:rgb({r},{g},{b});
                    border:1px solid rgba(0,0,0,0.3);
                    display:inline-block;
                "></span>
                <span><b>{label}</b> <span style="color:#666;">({value_range})</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Colors represent suitability score. When multiple crops are selected, "
        "the color uses the best score among the selected crops at that location."
    )


def prepare_selected_crop_view(
    df: pd.DataFrame,
    selected_crop_keys: list[str],
    profiles: dict,
    score_cols: dict[str, str],
) -> tuple[pd.DataFrame, str]:
    """
    Adds the map score/crop columns.

    - One crop selected: map score/color = that crop's score.
    - Multiple crops selected: map score/color = highest score among selected crops,
      and best_selected_crop identifies which crop won at each point.
    """
    out = df.copy()

    # Rounded score columns make PyDeck tooltips easier to read.
    for crop_key in selected_crop_keys:
        score_col = score_cols[crop_key]
        out[f"{crop_key}_display_score"] = out[score_col].round(1)

    if len(selected_crop_keys) == 1:
        crop_key = selected_crop_keys[0]
        score_col = score_cols[crop_key]

        out["map_score"] = out[score_col]
        out["map_score_display"] = out["map_score"].round(1)
        out["map_crop"] = crop_display_name(crop_key, profiles)
        out["map_mode"] = f"{crop_display_name(crop_key, profiles)} score"

    else:
        selected_score_cols = [score_cols[k] for k in selected_crop_keys]

        out["map_score"] = out[selected_score_cols].max(axis=1)
        out["map_score_display"] = out["map_score"].round(1)

        # idxmax returns the score column name; convert it back to the crop key/display name.
        col_to_crop = {score_cols[k]: k for k in selected_crop_keys}
        best_score_col = out[selected_score_cols].idxmax(axis=1)

        out["best_selected_crop_key"] = best_score_col.map(col_to_crop)
        out["map_crop"] = out["best_selected_crop_key"].map(
            lambda k: crop_display_name(k, profiles)
        )
        out["map_mode"] = "Best selected crop"

    out = add_color_column(out, "map_score")
    return out, "map_score"


def render_map(
    df: pd.DataFrame,
    selected_crop_keys: list[str],
    profiles: dict,
    score_cols: dict[str, str],
    radius: int = 120,
) -> None:
    required_cols = ["longitude", "latitude", "map_score", "map_crop", "color"]
    map_df = df.dropna(subset=required_cols).copy()

    if len(map_df) == 0:
        st.warning("No valid map rows found.")
        return

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

    crop_score_lines = ""
    for crop_key in selected_crop_keys:
        display = crop_display_name(crop_key, profiles)
        crop_score_lines += f"<br/><b>{display} score:</b> {{{crop_key}_display_score}}"

    tooltip = {
        "html": (
            "<b>Map mode:</b> {map_mode}"
            "<br/><b>Shown crop:</b> {map_crop}"
            "<br/><b>Shown score:</b> {map_score_display}"
            + crop_score_lines
            + "<br/><b>NDVI:</b> {NDVI}"
            "<br/><b>NDMI:</b> {NDMI}"
        ),
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=8,
            pitch=0,
        ),
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

    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top feature importance",
    )

    st.plotly_chart(fig, use_container_width=True)


st.title("Syria Crop Suitability PoC")
st.caption(
    "Screen 1: Olive and Damask rose suitability heatmaps from satellite, soil, climate, and terrain features."
)

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

    selected_crop_keys = st.multiselect(
        "Crop",
        crop_keys,
        default=[crop_keys[0]] if crop_keys else [],
        format_func=lambda k: crop_display_name(k, profiles),
        help=(
            "Select one crop to show that crop's suitability, or multiple crops "
            "to show the best selected crop at each location."
        ),
    )

    use_model = st.toggle("Use trained model score", value=True)
    min_score = st.slider("Minimum score", 0, 100, 0)
    radius = st.slider("Point radius", 30, 400, 120, step=10)

if not selected_crop_keys:
    st.warning("Select at least one crop in Map settings.")
    st.stop()

score_cols = {k: get_crop_score_column(k, use_model) for k in selected_crop_keys}

missing_score_cols = [c for c in score_cols.values() if c not in df.columns]

if missing_score_cols:
    st.error(
        f"Column(s) not found: {', '.join(missing_score_cols)}. "
        "Check prediction pipeline output."
    )
    st.stop()

prepared, map_score_col = prepare_selected_crop_view(
    df=df,
    selected_crop_keys=selected_crop_keys,
    profiles=profiles,
    score_cols=score_cols,
)

filtered = prepared[prepared[map_score_col] >= min_score].copy()

st.subheader("Selected crop suitability map")

if len(selected_crop_keys) == 1:
    crop_key = selected_crop_keys[0]
    crop_profile = profiles[crop_key]

    st.write(
        f"**Crop:** {crop_display_name(crop_key, profiles)}  |  "
        f"**Product:** {crop_profile.get('product', '')}  |  "
        f"**Sun:** {crop_profile.get('sun_requirement', '')}  |  "
        f"**Shade tolerance:** {crop_profile.get('shade_tolerance', '')}"
    )

    st.info(crop_profile.get("agrivoltaics_note", ""))

else:
    selected_names = [crop_display_name(k, profiles) for k in selected_crop_keys]

    st.write(
        "**Map mode:** Best selected crop per location  |  "
        f"**Selected crops:** {', '.join(selected_names)}"
    )

    st.info(
        "Each point is colored by the highest suitability score among the selected crops. "
        "The tooltip shows each selected crop's individual score."
    )

render_score_legend()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(filtered):,}")
c2.metric(
    "Mean shown score",
    f"{filtered[map_score_col].mean():.1f}" if len(filtered) else "n/a",
)
c3.metric("Priority zones", f"{(filtered[map_score_col] >= 80).sum():,}")
c4.metric("High+ zones", f"{(filtered[map_score_col] >= 65).sum():,}")

render_map(
    df=filtered,
    selected_crop_keys=selected_crop_keys,
    profiles=profiles,
    score_cols=score_cols,
    radius=radius,
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Score distribution")

    fig = px.histogram(
        filtered,
        x=map_score_col,
        nbins=30,
        title="Shown suitability score distribution",
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Feature importance")

    if len(selected_crop_keys) == 1:
        show_feature_importance(selected_crop_keys[0])

    else:
        tabs = st.tabs([crop_display_name(k, profiles) for k in selected_crop_keys])

        for tab, crop_key in zip(tabs, selected_crop_keys):
            with tab:
                show_feature_importance(crop_key)

st.subheader("Top candidate locations")

show_cols = ["longitude", "latitude", "map_crop", "map_score_display"]

for crop_key in selected_crop_keys:
    score_col = score_cols[crop_key]
    class_col = score_col.replace("score", "class")

    component_cols = [
        f"{crop_key}_soil_component",
        f"{crop_key}_water_component",
        f"{crop_key}_heat_component",
        f"{crop_key}_vegetation_component",
        f"{crop_key}_terrain_component",
    ]

    show_cols.append(score_col)

    if class_col in filtered.columns:
        show_cols.append(class_col)

    show_cols.extend(component_cols)

show_cols.extend(
    [
        "NDVI",
        "NDMI",
        "BSI",
        "soil_ph",
        "soil_org_carbon",
        "mean_temp_c",
        "annual_rain_mm",
        "slope_deg",
    ]
)

show_cols = [c for c in show_cols if c in filtered.columns]

top = filtered.sort_values(map_score_col, ascending=False).head(50).copy()

for crop_key in selected_crop_keys:
    score_col = score_cols[crop_key]
    class_col = score_col.replace("score", "class")

    if class_col in top.columns:
        top[f"{crop_key}_class_label"] = top[class_col].apply(class_label)
        show_cols.append(f"{crop_key}_class_label")

# Remove duplicates while preserving order.
show_cols = list(dict.fromkeys(show_cols))

st.dataframe(top[show_cols], use_container_width=True)

if len(selected_crop_keys) > 1:
    st.subheader("Best selected crop counts")

    counts = (
        filtered["map_crop"]
        .value_counts()
        .rename_axis("Best selected crop")
        .reset_index(name="Locations")
    )

    st.dataframe(counts, use_container_width=True)

st.subheader("Compare crops")

compare_cols = ["longitude", "latitude"]

for k in crop_keys:
    for suffix in ["model_score", "model_class", "rule_score", "rule_class"]:
        col = f"{k}_{suffix}"

        if col in df.columns:
            compare_cols.append(col)

for col in ["best_crop", "best_crop_score", "best_crop_class"]:
    if col in df.columns:
        compare_cols.append(col)

sort_col = "best_crop_score" if "best_crop_score" in df.columns else compare_cols[-1]

st.dataframe(
    df[compare_cols].sort_values(sort_col, ascending=False).head(100),
    use_container_width=True,
)

st.caption(
    "This dashboard is a screening tool. Final planting decisions require farmer validation and soil/lab testing."
)