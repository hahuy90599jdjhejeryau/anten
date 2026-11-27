# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from antenna_core import (
    MATERIALS,
    calculate_antenna,
    compare_materials,
    loop_field_pattern,
    make_sweep_data,
)
from ai_engine import generate_synthetic_dataset, predict_one, train_models

# -----------------------------------------------------------------------------
# AI model artifact configuration
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATHS = {
    "Random Forest": MODEL_DIR / "looplab_random_forest.joblib",
    "ANN": MODEL_DIR / "looplab_ann.joblib",
}
AI_INPUT_DOMAIN = {
    "radius_cm": (0.8, 8.0),
    "freq_mhz": (5.0, 50.0),
    "wire_radius_mm": (0.25, 1.5),
    "n_turns": (1, 4),
}
AI_FEATURES = ["radius_cm", "freq_mhz", "wire_radius_mm", "n_turns", "rho_ohm_m"]
AI_TARGETS = ["Rr_ohm", "Zin_real_ohm", "Zin_imag_ohm", "gain_dBi_approx"]
SESSION_BUNDLE_KEYS = {
    "Random Forest": "looplab_rf_bundle_v2",
    "ANN": "looplab_ann_bundle_v2",
}


def validate_model_bundle(bundle: Any, expected_name: str) -> dict[str, Any]:
    """Reject incompatible or incomplete model artifacts before prediction."""
    if not isinstance(bundle, dict):
        raise ValueError("Tệp mô hình không đúng định dạng bundle của LoopLab.")
    if bundle.get("model_name") != expected_name:
        raise ValueError(
            f"Tệp này được lưu cho {bundle.get('model_name')!r}, không phải {expected_name!r}."
        )
    if list(bundle.get("feature_columns", [])) != AI_FEATURES:
        raise ValueError("Danh sách đầu vào trong mô hình không tương thích với phiên bản LoopLab này.")
    models = bundle.get("models")
    if not isinstance(models, dict) or any(target not in models for target in AI_TARGETS):
        raise ValueError("Bundle không chứa đủ 4 mô hình đầu ra cần thiết.")
    if not isinstance(bundle.get("metrics"), pd.DataFrame):
        raise ValueError("Bundle không chứa bảng metrics hợp lệ.")
    return bundle


def serialize_bundle(bundle: dict[str, Any]) -> bytes:
    """Serialize a trained bundle for download or repository storage."""
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def load_bundle_bytes(raw: bytes, expected_name: str) -> dict[str, Any]:
    """Load only a user-owned LoopLab artifact; joblib files are pickle-based."""
    bundle = joblib.load(io.BytesIO(raw))
    return validate_model_bundle(bundle, expected_name)


def load_saved_bundles_once() -> list[str]:
    """Load committed/local model files into this Streamlit session once."""
    messages: list[str] = []
    for model_name, path in MODEL_PATHS.items():
        state_key = SESSION_BUNDLE_KEYS[model_name]
        if state_key in st.session_state or not path.is_file():
            continue
        try:
            bundle = joblib.load(path)
            st.session_state[state_key] = validate_model_bundle(bundle, model_name)
            messages.append(f"Đã nạp {model_name} từ {path.relative_to(APP_DIR)}.")
        except Exception as exc:
            messages.append(f"Không nạp được {path.name}: {exc}")
    return messages


def save_bundle_locally(bundle: dict[str, Any], model_name: str) -> str | None:
    """Save a model beside the app; on Streamlit Cloud this filesystem may be ephemeral."""
    path = MODEL_PATHS[model_name]
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path, compress=3)
        return str(path.relative_to(APP_DIR))
    except OSError:
        return None


def ai_domain_violations(radius: float, freq: float, wire: float, turns: int, material_name: str) -> list[str]:
    """Return explicit reasons why an input is outside the synthetic training domain."""
    checks = [
        ("radius_cm", radius, "Bán kính R"),
        ("freq_mhz", freq, "Tần số f"),
        ("wire_radius_mm", wire, "Bán kính dây a"),
    ]
    problems: list[str] = []
    for key, value, label in checks:
        low, high = AI_INPUT_DOMAIN[key]
        if not (low <= float(value) <= high):
            problems.append(f"{label} phải nằm trong [{low:g}, {high:g}].")
    low_n, high_n = AI_INPUT_DOMAIN["n_turns"]
    if int(turns) != turns or not (low_n <= int(turns) <= high_n):
        problems.append(f"Số vòng N phải là số nguyên trong [{low_n}, {high_n}].")
    if material_name not in MATERIALS:
        problems.append("Vật liệu chưa có trong bộ dữ liệu huấn luyện.")
    # Keep the physical small-loop criterion explicit, even though the current
    # generated range is designed to remain below 0.1 at its upper corner.
    circumference_lambda = 2.0 * math.pi * (float(radius) / 100.0) * (float(freq) * 1e6) / 299_792_458.0
    if circumference_lambda >= 0.1:
        problems.append(f"C/λ = {circumference_lambda:.5g} không thỏa điều kiện vòng nhỏ C/λ < 0.1.")
    return problems


def build_model_comparison(rf_bundle: dict[str, Any], ann_bundle: dict[str, Any]) -> pd.DataFrame:
    """Merge metrics from models evaluated on the same held-out split."""
    rf = rf_bundle["metrics"].set_index("Đại lượng")
    ann = ann_bundle["metrics"].set_index("Đại lượng")
    rows = []
    for target in AI_TARGETS:
        rf_row = rf.loc[target]
        ann_row = ann.loc[target]
        rows.append({
            "Đầu ra": target,
            "RF · MAE": float(rf_row["MAE"]),
            "ANN · MAE": float(ann_row["MAE"]),
            "RF · RMSE": float(rf_row["RMSE"]),
            "ANN · RMSE": float(ann_row["RMSE"]),
            "RF · R²": float(rf_row["R²"]),
            "ANN · R²": float(ann_row["R²"]),
            "MAE tốt hơn": "Random Forest" if float(rf_row["MAE"]) <= float(ann_row["MAE"]) else "ANN",
            "RMSE tốt hơn": "Random Forest" if float(rf_row["RMSE"]) <= float(ann_row["RMSE"]) else "ANN",
            "R² tốt hơn": "Random Forest" if float(rf_row["R²"]) >= float(ann_row["R²"]) else "ANN",
        })
    return pd.DataFrame(rows)


def build_prediction_comparison(
    base: dict[str, Any], values: dict[str, float], rf_bundle: dict[str, Any], ann_bundle: dict[str, Any]
) -> pd.DataFrame:
    """Predict with both saved models and compare against the analytical calculation."""
    rf_prediction = predict_one(rf_bundle, values)
    ann_prediction = predict_one(ann_bundle, values)
    source_rows = [
        ("Rr (Ω)", "Rr_ohm", float(base["Rr_ohm"])),
        ("Re(Zin) (Ω)", "Zin_real_ohm", float(base["Zin_real_ohm"])),
        ("Im(Zin) (Ω)", "Zin_imag_ohm", float(base["Zin_imag_ohm"])),
        ("Gain (dBi)", "gain_dBi_approx", float(base["gain_dBi_approx"])),
    ]
    result = []
    for label, target, analytical in source_rows:
        rf_value = float(rf_prediction[target])
        ann_value = float(ann_prediction[target])
        scale = max(abs(analytical), 1e-12)
        result.append({
            "Đại lượng": label,
            "Mô hình giải tích": analytical,
            "Random Forest": rf_value,
            "Sai số RF (%)": abs(rf_value - analytical) / scale * 100.0,
            "ANN": ann_value,
            "Sai số ANN (%)": abs(ann_value - analytical) / scale * 100.0,
        })
    return pd.DataFrame(result)

st.set_page_config(
    page_title="LoopLab | Mô phỏng anten khung tròn nhỏ",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 80% 0%, rgba(34,112,171,.08), transparent 28%), #F3F7FB;
}
[data-testid="stHeader"] { background: rgba(243,247,251,.88); }
.block-container { max-width: 1540px; padding-top: 1.15rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10283E 0%,#173B59 100%); }
[data-testid="stSidebar"] * { color: #F3F7FB !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: #F8FBFE !important; color: #17324D !important; border-radius: 9px;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input { color: #17324D !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #E4EEF7 !important; font-weight: 600; }
.hero {
  padding: 1.55rem 1.8rem; border-radius: 22px; color: #F8FBFF;
  background: linear-gradient(115deg,#10283E 0%,#174A70 58%,#207D91 100%);
  box-shadow: 0 14px 34px rgba(16,40,62,.16); margin-bottom: 1.1rem;
  position: relative; overflow: hidden;
}
.hero:after { content:""; position:absolute; width:280px; height:280px; border:1px solid rgba(255,255,255,.15);
  border-radius:50%; right:-70px; top:-130px; box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
.hero-kicker { font-size:.74rem; letter-spacing:.16em; font-weight:700; color:#9FD5E3; text-transform:uppercase; }
.hero h1 { font-size:2.0rem; line-height:1.2; margin:.25rem 0 .45rem 0; color:#FFFFFF; font-weight:800; }
.hero p { color:#D7E8F4; margin:0; font-size:.94rem; max-width:900px; }
.section-title { color:#17324D; font-size:1.12rem; font-weight:800; margin:.3rem 0 .65rem 0; }
.section-caption { color:#6A7E90; font-size:.84rem; margin-top:-.35rem; margin-bottom:.8rem; }
.kpi-card { background:#FFFFFF; border:1px solid #E0E8EF; border-radius:15px; padding: .9rem 1rem;
  box-shadow:0 5px 18px rgba(28,57,82,.055); min-height:112px; }
.kpi-label { color:#718397; font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.kpi-value { color:#17324D; font-size:1.28rem; font-weight:800; margin-top:.36rem; overflow-wrap:anywhere; }
.kpi-foot { color:#8292A2; font-size:.72rem; margin-top:.18rem; }
.info-card { background:#FFFFFF; border:1px solid #DFE8F0; border-radius:15px; padding:1rem 1.1rem; }
.note-box { background:#EAF3F8; border-left:4px solid #2586A0; padding:.85rem 1rem; border-radius:9px; color:#31536A; font-size:.86rem; }
.warning-box { background:#FFF5E8; border-left:4px solid #E7A344; padding:.85rem 1rem; border-radius:9px; color:#76511E; font-size:.86rem; }
.good-box { background:#EAF7F1; border-left:4px solid #2D9B70; padding:.85rem 1rem; border-radius:9px; color:#245D48; font-size:.86rem; }
.small-muted { color:#74879A; font-size:.77rem; }
div[data-testid="stTabs"] button { font-weight:700; color:#50677D; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#1769AA; }
.stDownloadButton button, .stButton button { border-radius:10px; font-weight:700; }
/* Ensure primary action labels remain legible on Streamlit's blue button. */
div[data-testid="stButton"] button[kind="primary"],
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  color: #FFFFFF !important;
}
div[data-testid="stButton"] button[kind="primary"] *,
.stButton > button[kind="primary"] * ,
[data-testid="stBaseButton-primary"] * {
  color: #FFFFFF !important;
}
hr { border-color:#E0E8EF; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_base_and_sweeps(
    radius_cm: float,
    freq_mhz: float,
    wire_radius_mm: float,
    n_turns: int,
    z0_ohm: float,
    material: str,
):
    base = calculate_antenna(radius_cm, freq_mhz, wire_radius_mm, n_turns, z0_ohm, material)
    sweeps = make_sweep_data(base)
    return base, sweeps


def fmt(value: float, digits: int = 4) -> str:
    try:
        x = float(value)
        if not math.isfinite(x):
            return "—"
        if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
            return f"{x:.{digits-1}e}"
        return f"{x:,.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def kpi(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        # Reserve extra top space so the title and horizontal legend do not overlap.
        margin=dict(l=12, r=12, t=68, b=12),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Be Vietnam Pro, sans-serif", color="#34495E", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#17324D", font_color="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8EEF3", zerolinecolor="#D8E2EA")
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF3", zerolinecolor="#D8E2EA")
    return fig


def antenna_geometry_figure(base: dict[str, Any]) -> go.Figure:
    radius = float(base["radius_cm"])
    turns = int(base["n_turns"])
    material = str(base["material"])
    metal_color = str(MATERIALS[material]["color"])
    gap = math.radians(10)
    theta = np.linspace(gap / 2, 2 * math.pi - gap / 2, 480)
    visible_turns = min(turns, 8)
    spacing = max(0.035 * radius, 0.08)
    z_offsets = (np.arange(visible_turns) - (visible_turns - 1) / 2) * spacing
    fig = go.Figure()

    for index, zoff in enumerate(z_offsets):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.full_like(theta, zoff)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color=metal_color, width=9),
            name=f"Vòng dây {index + 1}",
            hovertemplate="x=%{x:.2f} cm<br>y=%{y:.2f} cm<extra>Dây dẫn</extra>",
        ))

    # Feed terminals and short feed lines.
    p1 = (radius * math.cos(gap / 2), radius * math.sin(gap / 2), 0.0)
    p2 = (radius * math.cos(2 * math.pi - gap / 2), radius * math.sin(2 * math.pi - gap / 2), 0.0)
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode="markers", marker=dict(size=6, color=["#E7B84D", "#95A3B3"], line=dict(color="white", width=1)),
        name="Hai đầu cấp nguồn", hoverinfo="name",
    ))
    feed_z = -0.36 * radius
    fig.add_trace(go.Scatter3d(
        x=[p1[0], 0.76 * radius, 0.76 * radius, p2[0]],
        y=[p1[1], 0.08 * radius, -0.08 * radius, p2[1]],
        z=[0, feed_z, feed_z, 0], mode="lines",
        line=dict(color="#536579", width=5), name="Đường cấp nguồn minh họa",
    ))
    # Radius reference line.
    fig.add_trace(go.Scatter3d(
        x=[0, radius], y=[0, 0], z=[0, 0], mode="lines+text",
        line=dict(color="#2479B5", width=4, dash="dash"),
        text=["", f"R = {radius:g} cm"], textposition="top center",
        name="Bán kính R",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=4, color="#17324D"), name="Tâm vòng",
    ))
    fig.update_layout(
        title=dict(text=f"MÔ HÌNH HÌNH HỌC 3D · {material} · {turns} vòng", x=0.02),
        scene=dict(
            xaxis_title="X (cm)", yaxis_title="Y (cm)", zaxis_title="Z (cm)",
            aspectmode="data", camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
            xaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
            yaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
            zaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
        ),
        height=550,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", color="#34495E"),
        legend=dict(orientation="h", y=-0.02),
    )
    if turns > visible_turns:
        fig.add_annotation(
            text=f"Hiển thị {visible_turns}/{turns} vòng để mô hình dễ quan sát.",
            x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
            bgcolor="rgba(255,255,255,.88)", font=dict(color="#8A5A16", size=11),
        )
    return fig


def radiation_figure(base: dict[str, Any]) -> go.Figure:
    theta = np.linspace(0, np.pi, 90)
    phi = np.linspace(0, 2 * np.pi, 150)
    th, ph = np.meshgrid(theta, phi)
    field = loop_field_pattern(th, base)
    power = field**2
    x = field * np.sin(th) * np.cos(ph)
    y = field * np.sin(th) * np.sin(ph)
    z = field * np.cos(th)
    fig = go.Figure(go.Surface(
        x=x, y=y, z=z, surfacecolor=power,
        colorscale="Turbo", cmin=0, cmax=1,
        colorbar=dict(title="P/Pmax", thickness=14, len=0.68),
        hovertemplate="P/Pmax=%{surfacecolor:.3f}<extra>Pattern chuẩn hóa</extra>",
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text="GIẢN ĐỒ BỨC XẠ 3D · CÔNG SUẤT CHUẨN HÓA", x=0.02),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="cube", camera=dict(eye=dict(x=1.45, y=1.45, z=1.05)),
            xaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
            yaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
            zaxis=dict(backgroundcolor="#F7FAFC", gridcolor="#E5EDF3"),
        ),
        height=550, margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", color="#34495E"),
    )
    return fig


def polar_pattern_figure(base: dict[str, Any]) -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 1000)
    field = loop_field_pattern(theta, base)
    power_db = 10 * np.log10(np.maximum(field**2, 1e-4))
    radius = np.clip(power_db, -40, 0) + 40
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        theta=np.degrees(theta), r=radius, mode="lines",
        line=dict(color="#169A9A", width=3), fill="toself",
        fillcolor="rgba(22,154,154,.10)", name="Công suất chuẩn hóa",
        hovertemplate="Góc=%{theta:.1f}°<br>Mức tương đối=%{r:.1f} dB<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="GIẢN ĐỒ CÔNG SUẤT 2D", x=0.02),
        polar=dict(
            bgcolor="#FFFFFF", angularaxis=dict(rotation=90, direction="clockwise", gridcolor="#E4EBF1"),
            radialaxis=dict(range=[0, 40], tickvals=[0, 10, 20, 30, 40],
                            ticktext=["-40 dB", "-30", "-20", "-10", "0 dB"], gridcolor="#E4EBF1"),
        ),
        height=430, margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", color="#34495E"), showlegend=False,
    )
    return fig


def comparison_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        ("Vật liệu", "material"),
        ("Điện trở suất (Ω·m)", "rho_ohm_m"),
        ("Rac (Ω)", "Rac_ohm"),
        ("Rr (Ω)", "Rr_ohm"),
        ("Hiệu suất (%)", "radiation_efficiency_pct"),
        ("Gain (dBi)", "gain_dBi_approx"),
        ("Re(Zin) (Ω)", "Zin_real_ohm"),
        ("Im(Zin) (Ω)", "Zin_imag_ohm"),
        ("Khối lượng dây (g)", "wire_mass_g"),
    ]
    return pd.DataFrame([{label: row[key] for label, key in columns} for row in rows])


# -----------------------------------------------------------------------------
# Sidebar: inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:1.1rem;font-weight:800;letter-spacing:.08em;'>📡 LOOPLAB</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.78rem;color:#B9D1E2;margin-bottom:1.1rem;'>SMALL CIRCULAR LOOP · CHƯƠNG 5.6</div>", unsafe_allow_html=True)
    st.markdown("### Thông số thiết kế")
    radius_cm = st.number_input("Bán kính vòng R (cm)", min_value=0.05, max_value=200.0, value=5.0, step=0.5)
    freq_mhz = st.number_input("Tần số làm việc f (MHz)", min_value=0.1, max_value=1000.0, value=30.0, step=1.0)
    wire_radius_mm = st.number_input("Bán kính dây dẫn a (mm)", min_value=0.05, max_value=20.0, value=1.0, step=0.1)
    n_turns = st.number_input("Số vòng dây N", min_value=1, max_value=30, value=1, step=1)
    material = st.selectbox("Vật liệu dây dẫn", list(MATERIALS.keys()), index=0)
    z0_ohm = st.number_input("Trở kháng tham chiếu Z0 (Ω)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
    st.markdown("---")
    st.markdown("**Gợi ý sử dụng**")
    st.caption("Thay đổi R, f, bán kính dây, số vòng hoặc vật liệu. Các chỉ số và đồ thị được cập nhật theo bộ tham số mới.")
    st.caption("Dây dẫn được coi là đồng nhất, nhiệt độ tham chiếu xấp xỉ 20 °C.")

# -----------------------------------------------------------------------------
# Compute selected point and sweeps
# -----------------------------------------------------------------------------
try:
    base, sweep_data = cached_base_and_sweeps(
        float(radius_cm), float(freq_mhz), float(wire_radius_mm), int(n_turns), float(z0_ohm), material
    )
    radius_rows, freq_rows, heat_radii, heat_freqs, heat_rr_db, heat_gain_db = sweep_data
except (ValueError, OverflowError, FloatingPointError) as exc:
    st.markdown(
        "<div class='hero'><div class='hero-kicker'>LOOPLAB · SMALL LOOP ANTENNA</div>"
        "<h1>Mô phỏng anten khung tròn nhỏ</h1><p>Nhập thông số hợp lệ để bắt đầu mô phỏng.</p></div>",
        unsafe_allow_html=True,
    )
    st.error(str(exc))
    st.info("Kiểm tra lại tỉ lệ giữa bán kính vòng R và bán kính dây dẫn a. Công thức điện cảm gần đúng yêu cầu 8R/a > e² (đổi về cùng đơn vị SI).")
    st.stop()

# Hero
st.markdown(
    "<div class='hero'>"
    "<div class='hero-kicker'>ENGINEERING SIMULATION · CHAPTER 5.6</div>"
    "<h1>Small Circular Loop Antenna</h1>"
    "<p>Mô phỏng hình học 3D, đặc tính bức xạ, trở kháng đầu vào, khảo sát tham số và mô hình AI dự đoán. "
    "Thay đổi vật liệu dây dẫn để quan sát tác động của tổn hao dẫn điện.</p>"
    "</div>",
    unsafe_allow_html=True,
)

status_text = "ĐẠT TIÊU CHÍ VÒNG NHỎ" if base["is_small_loop"] else "VƯỢT NGƯỠNG VÒNG NHỎ"
status_html = "good-box" if base["is_small_loop"] else "warning-box"
st.markdown(
    f"<div class='{status_html}'><b>{status_text}</b> · C/λ = {base['circumference_lambda']:.5g} "
    f"{'<' if base['is_small_loop'] else '≥'} 0.1. "
    f"{'Điểm làm việc nằm trong miền vòng nhỏ theo tiêu chí đang dùng.' if base['is_small_loop'] else 'Kết quả giải tích có thể kém chính xác khi kích thước điện vượt miền vòng nhỏ.'}</div>",
    unsafe_allow_html=True,
)

# KPI band
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi("Điện trở bức xạ · Rr", f"{fmt(base['Rr_ohm'], 4)} Ω", "Theo mô hình vòng nhỏ")
with kpi_cols[1]:
    kpi("Trở kháng đầu vào · |Zin|", f"{fmt(base['Zin_magnitude_ohm'], 4)} Ω", f"{fmt(base['Zin_real_ohm'], 4)} + j{fmt(base['Zin_imag_ohm'], 4)} Ω")
with kpi_cols[2]:
    kpi("Gain xấp xỉ", f"{fmt(base['gain_dBi_approx'], 4)} dBi", "Chưa tính mismatch")
with kpi_cols[3]:
    kpi("Hiệu suất bức xạ", f"{fmt(base['radiation_efficiency_pct'], 4)} %", f"Vật liệu: {material}")

st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

# Top-level navigation tabs
(tab_overview, tab_3d, tab_sweep, tab_material, tab_ai, tab_theory) = st.tabs([
    "◈  Tổng quan", "◉  Mô hình 3D", "⌁  Khảo sát R–f", "◆  So sánh vật liệu", "✦  AI dự đoán", "ⓘ  Lý thuyết & hướng dẫn"
])

with tab_overview:
    section("Bảng điều khiển mô phỏng", "Tóm tắt điểm làm việc hiện tại và các thông số điện quan trọng.")
    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(antenna_geometry_figure(base), width="stretch", key="overview_geometry")
    with right:
        st.plotly_chart(radiation_figure(base), width="stretch", key="overview_radiation")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(polar_pattern_figure(base), width="stretch", key="overview_polar")
    with col2:
        radius_values = np.array([row["radius_cm"] for row in radius_rows], dtype=float)
        rr_values = np.array([row["Rr_ohm"] for row in radius_rows], dtype=float)
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Scatter(x=radius_values, y=rr_values, mode="lines", name="Rr",
                                    line=dict(color="#2479B5", width=3), fill="tozeroy",
                                    fillcolor="rgba(36,121,181,.09)"))
        fig_rr.add_vline(x=radius_cm, line_dash="dash", line_color="#E79A37", annotation_text="R hiện tại")
        fig_rr.update_yaxes(type="log", title="Rr (Ω, log)")
        fig_rr.update_xaxes(title="Bán kính R (cm)")
        fig_rr.update_layout(title="Điện trở bức xạ theo bán kính", height=430)
        st.plotly_chart(plot_layout(fig_rr, 430), width="stretch", key="overview_rr")

    section("Thông số điện tại điểm làm việc")
    details = pd.DataFrame([
        {"Đại lượng": "Bước sóng λ", "Giá trị": f"{fmt(base['wavelength_m'], 6)} m", "Diễn giải": "λ = c/f"},
        {"Đại lượng": "Chu vi / bước sóng", "Giá trị": fmt(base["circumference_lambda"], 6), "Diễn giải": "Điều kiện anten vòng nhỏ: C/λ < 0.1"},
        {"Đại lượng": "Điện cảm L", "Giá trị": f"{fmt(base['L_uH'], 6)} µH", "Diễn giải": "Gần đúng; N vòng dùng L ≈ N²L₁"},
        {"Đại lượng": "Độ sâu bề mặt δ", "Giá trị": f"{fmt(base['skin_depth_um'], 6)} µm", "Diễn giải": "Phụ thuộc tần số và điện trở suất vật liệu"},
        {"Đại lượng": "Điện trở DC Rdc", "Giá trị": f"{fmt(base['Rdc_ohm'], 6)} Ω", "Diễn giải": "Từ điện trở suất và kích thước dây"},
        {"Đại lượng": "Điện trở AC Rac", "Giá trị": f"{fmt(base['Rac_ohm'], 6)} Ω", "Diễn giải": "Gần đúng có xét hiệu ứng bề mặt"},
        {"Đại lượng": "Pha trở kháng", "Giá trị": f"{fmt(base['Zin_phase_deg'], 5)}°", "Diễn giải": "Góc pha của Zin"},
        {"Đại lượng": "SWR tại Z0", "Giá trị": fmt(base["swr_z0_approx"], 5), "Diễn giải": "Ước tính từ hệ số phản xạ"},
        {"Đại lượng": "Realized gain", "Giá trị": f"{fmt(base['realized_gain_dBi_approx'], 5)} dBi", "Diễn giải": "Gain có tính mismatch theo Z0"},
        {"Đại lượng": "Khối lượng dây ước tính", "Giá trị": f"{fmt(base['wire_mass_g'], 5)} g", "Diễn giải": "Không tính lớp mạ, mối nối và giá đỡ"},
    ])
    st.dataframe(details, hide_index=True, width="stretch")
    result_csv = pd.DataFrame([base]).to_csv(index=False).encode("utf-8-sig")
    st.download_button("↓ Tải thông số điểm làm việc (CSV)", result_csv, "thong_so_diem_lam_viec.csv", "text/csv")

with tab_3d:
    section("Mô hình hình học và giản đồ bức xạ", "Xoay, phóng to và quan sát trực tiếp bằng chuột; các biểu đồ sử dụng kích thước và tần số hiện tại.")
    geom_col, pattern_col = st.columns(2)
    with geom_col:
        st.plotly_chart(antenna_geometry_figure(base), width="stretch", key="full_geometry")
    with pattern_col:
        st.plotly_chart(radiation_figure(base), width="stretch", key="full_radiation")
    st.plotly_chart(polar_pattern_figure(base), width="stretch", key="full_polar")
    st.markdown(
        "<div class='note-box'><b>Ghi chú mô hình:</b> Hình học 3D biểu diễn vòng dây, khe cấp nguồn và đường cấp nguồn ở mức trực quan. "
        "Giản đồ bức xạ dùng phân bố trường xa chuẩn hóa theo hàm Bessel J₁ cho vòng dòng đều. Đây không phải kết quả giải Maxwell/full-wave; "
        "dây và phụ kiện trên hình không được mô phỏng như các vật thể điện từ độc lập.</div>",
        unsafe_allow_html=True,
    )

with tab_sweep:
    section("Khảo sát ảnh hưởng của bán kính và tần số", "Các đường cong và bản đồ nhiệt được tính lại từ mô hình giải tích khi thông số đầu vào thay đổi.")
    sweep_a, sweep_b = st.columns(2)
    with sweep_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["radius_cm"] for r in radius_rows], y=[r["Rr_ohm"] for r in radius_rows],
                                 name="Rr", mode="lines", line=dict(color="#2479B5", width=3)))
        fig.add_vline(x=radius_cm, line_dash="dash", line_color="#E79A37")
        fig.update_xaxes(title="R (cm)")
        fig.update_yaxes(title="Rr (Ω)", type="log")
        fig.update_layout(title=f"Rr theo bán kính · f = {freq_mhz:g} MHz", height=360)
        st.plotly_chart(plot_layout(fig, 360), width="stretch")
    with sweep_b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["freq_mhz"] for r in freq_rows], y=[r["Rr_ohm"] for r in freq_rows],
                                 name="Rr", mode="lines", line=dict(color="#2479B5", width=3)))
        fig.add_vline(x=freq_mhz, line_dash="dash", line_color="#E79A37")
        fig.update_xaxes(title="f (MHz)")
        fig.update_yaxes(title="Rr (Ω)", type="log")
        fig.update_layout(title=f"Rr theo tần số · R = {radius_cm:g} cm", height=360)
        st.plotly_chart(plot_layout(fig, 360), width="stretch")

    fig_gain = go.Figure()
    fig_gain.add_trace(go.Scatter(x=[r["radius_cm"] for r in radius_rows], y=[r["gain_dBi_approx"] for r in radius_rows],
                                  name="Gain", line=dict(color="#7A5195", width=3)))
    fig_gain.add_trace(go.Scatter(x=[r["radius_cm"] for r in radius_rows], y=[r["radiation_efficiency_pct"] for r in radius_rows],
                                  name="Hiệu suất (%)", yaxis="y2", line=dict(color="#159A9C", width=2, dash="dot")))
    fig_gain.update_layout(title="Gain và hiệu suất theo bán kính", height=380,
                           xaxis_title="R (cm)", yaxis=dict(title="Gain (dBi)"),
                           yaxis2=dict(title="Hiệu suất (%)", overlaying="y", side="right"))
    st.plotly_chart(plot_layout(fig_gain, 380), width="stretch")

    heat_a, heat_b = st.columns(2)
    with heat_a:
        fig = go.Figure(go.Heatmap(
            x=heat_radii, y=heat_freqs, z=heat_rr_db, colorscale="Viridis",
            colorbar=dict(title="10log₁₀ Rr"), hovertemplate="R=%{x:.3f} cm<br>f=%{y:.3f} MHz<br>10log₁₀Rr=%{z:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(x=[radius_cm], y=[freq_mhz], mode="markers", name="Điểm hiện tại",
                                 marker=dict(color="white", size=11, symbol="x", line=dict(width=2))))
        fig.update_layout(title="Bản đồ nhiệt điện trở bức xạ", xaxis_title="R (cm)", yaxis_title="f (MHz)", height=420)
        st.plotly_chart(plot_layout(fig, 420), width="stretch")
    with heat_b:
        fig = go.Figure(go.Heatmap(
            x=heat_radii, y=heat_freqs, z=heat_gain_db, colorscale="Plasma",
            colorbar=dict(title="Gain (dBi)"), hovertemplate="R=%{x:.3f} cm<br>f=%{y:.3f} MHz<br>Gain=%{z:.3f} dBi<extra></extra>"
        ))
        fig.add_trace(go.Scatter(x=[radius_cm], y=[freq_mhz], mode="markers", name="Điểm hiện tại",
                                 marker=dict(color="white", size=11, symbol="x", line=dict(width=2))))
        fig.update_layout(title="Bản đồ nhiệt Gain ước tính", xaxis_title="R (cm)", yaxis_title="f (MHz)", height=420)
        st.plotly_chart(plot_layout(fig, 420), width="stretch")

    export_rows = pd.DataFrame(radius_rows)
    st.download_button("↓ Tải dữ liệu quét bán kính (CSV)", export_rows.to_csv(index=False).encode("utf-8-sig"),
                       "quet_ban_kinh.csv", "text/csv")
    st.download_button("↓ Tải dữ liệu quét tần số (CSV)", pd.DataFrame(freq_rows).to_csv(index=False).encode("utf-8-sig"),
                       "quet_tan_so.csv", "text/csv")

with tab_material:
    section("So sánh vật liệu dây dẫn", "Giữ nguyên hình học và tần số, chỉ thay đổi điện trở suất và mật độ khối của vật liệu.")
    material_rows = compare_materials(radius_cm, freq_mhz, wire_radius_mm, int(n_turns), z0_ohm)
    material_df = comparison_dataframe(material_rows)
    st.dataframe(material_df, hide_index=True, width="stretch", column_config={
        "Điện trở suất (Ω·m)": st.column_config.NumberColumn(format="%.3e"),
        "Rac (Ω)": st.column_config.NumberColumn(format="%.6g"),
        "Rr (Ω)": st.column_config.NumberColumn(format="%.6g"),
        "Hiệu suất (%)": st.column_config.NumberColumn(format="%.5f"),
        "Gain (dBi)": st.column_config.NumberColumn(format="%.5f"),
        "Re(Zin) (Ω)": st.column_config.NumberColumn(format="%.6g"),
        "Im(Zin) (Ω)": st.column_config.NumberColumn(format="%.6g"),
        "Khối lượng dây (g)": st.column_config.NumberColumn(format="%.5f"),
    })

    bar1, bar2 = st.columns(2)
    with bar1:
        fig = go.Figure(go.Bar(
            x=[r["material"] for r in material_rows], y=[r["Rac_ohm"] for r in material_rows],
            marker_color=[MATERIALS[r["material"]]["color"] for r in material_rows],
            text=[fmt(r["Rac_ohm"], 4) for r in material_rows], textposition="outside",
            hovertemplate="%{x}<br>Rac=%{y:.6g} Ω<extra></extra>",
        ))
        fig.update_layout(title="Tổn hao dẫn điện · Rac", yaxis_title="Rac (Ω)", height=390)
        st.plotly_chart(plot_layout(fig, 390), width="stretch")
    with bar2:
        fig = go.Figure(go.Bar(
            x=[r["material"] for r in material_rows], y=[r["gain_dBi_approx"] for r in material_rows],
            marker_color=[MATERIALS[r["material"]]["color"] for r in material_rows],
            text=[f"{r['gain_dBi_approx']:.3f}" for r in material_rows], textposition="outside",
            hovertemplate="%{x}<br>Gain=%{y:.4f} dBi<extra></extra>",
        ))
        fig.update_layout(title="Gain xấp xỉ theo vật liệu", yaxis_title="Gain (dBi)", height=390)
        st.plotly_chart(plot_layout(fig, 390), width="stretch")

    selected_row = next(row for row in material_rows if row["material"] == material)
    best_conductor = min(material_rows, key=lambda row: row["Rac_ohm"])
    st.markdown(
        f"<div class='note-box'><b>Nhận xét tại bộ thông số hiện tại:</b> {best_conductor['material']} có Rac thấp nhất "
        f"({fmt(best_conductor['Rac_ohm'], 5)} Ω) trong bốn vật liệu được xét. Vật liệu đang chọn là <b>{material}</b>, "
        f"Rac = {fmt(selected_row['Rac_ohm'], 5)} Ω, hiệu suất = {fmt(selected_row['radiation_efficiency_pct'], 5)}%. "
        "Trong mô hình này, Rr và giản đồ bức xạ chuẩn hóa không đổi khi chỉ đổi vật liệu; khác biệt đi vào qua tổn hao dẫn điện Rac, "
        "do đó ảnh hưởng đến hiệu suất và Gain.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='warning-box'><b>Lưu ý kỹ thuật:</b> Bảng vật liệu dùng điện trở suất khối gần đúng ở khoảng 20 °C. "
        "Mô hình chưa xét độ nhám bề mặt, lớp mạ, oxide nhôm, chất lượng mối nối, độ tinh khiết, nhiệt độ tăng do công suất RF "
        "và hiệu ứng lân cận giữa các vòng dây. Khối lượng chỉ là ước tính của phần dây dẫn.</div>",
        unsafe_allow_html=True,
    )

with tab_ai:
    section(
        "Mô hình AI dự đoán đặc tính anten",
        "Huấn luyện đồng thời Random Forest và ANN trên cùng một dataset, so sánh công bằng, lưu mô hình độc lập và kiểm soát miền dự đoán."
    )
    st.markdown(
        "<div class='note-box'><b>Quy trình:</b> Sinh một dataset duy nhất → dùng cùng chỉ số train/test 80/20 cho cả hai thuật toán → huấn luyện RF và ANN → so sánh MAE, RMSE, R² → lưu hai mô hình → dự đoán điểm làm việc hiện tại.</div>",
        unsafe_allow_html=True,
    )

    # Try to load previously saved artifacts from ./models on the first run.
    load_messages = load_saved_bundles_once()
    for message in load_messages:
        if message.startswith("Đã nạp"):
            st.success(message)
        else:
            st.warning(message)

    ai_left, ai_right = st.columns([1.05, 1.0])
    with ai_left:
        st.markdown("**Thiết lập huấn luyện**")
        sample_count = st.slider(
            "Số mẫu dữ liệu tổng hợp", min_value=500, max_value=6000,
            value=2500, step=500, key="ai_sample_count_v2"
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42,
            step=1, key="ai_seed_v2"
        )
        train_both_clicked = st.button(
            "⚙ Huấn luyện đồng thời RF + ANN", type="primary", width="stretch",
            key="train_both_ai_v2"
        )
    with ai_right:
        st.markdown("**Đầu vào mô hình**")
        st.write("R · f · a · N · điện trở suất vật liệu ρ")
        st.markdown("**Đầu ra dự đoán**")
        st.write("Rr · Re(Zin) · Im(Zin) · Gain")
        st.markdown("**Đánh giá**")
        st.write("MAE · RMSE · R² trên cùng tập test độc lập")
        st.caption("Cả hai mô hình nhận cùng dataset và cùng phép chia train/test có seed giống nhau.")

    if train_both_clicked:
        try:
            with st.spinner("Đang sinh dataset và huấn luyện cả Random Forest lẫn ANN..."):
                generated_dataset = generate_synthetic_dataset(int(sample_count), int(seed))
                rf_bundle = train_models(generated_dataset, model_name="Random Forest", random_state=int(seed))
                ann_bundle = train_models(generated_dataset, model_name="ANN", random_state=int(seed))

                # Attach provenance so the exported artifacts are self-describing.
                training_metadata = {
                    "dataset_rows": int(len(generated_dataset)),
                    "n_train": int(rf_bundle["n_train"]),
                    "n_test": int(rf_bundle["n_test"]),
                    "random_state": int(seed),
                    "train_test_ratio": "80/20",
                    "input_domain": AI_INPUT_DOMAIN.copy(),
                    "materials": list(MATERIALS.keys()),
                    "target_definition": "Synthetic labels generated by antenna_core.calculate_antenna",
                }
                rf_bundle["training_metadata"] = training_metadata
                ann_bundle["training_metadata"] = training_metadata

            st.session_state[SESSION_BUNDLE_KEYS["Random Forest"]] = rf_bundle
            st.session_state[SESSION_BUNDLE_KEYS["ANN"]] = ann_bundle
            st.session_state["looplab_ai_dataset_v2"] = generated_dataset
            rf_saved = save_bundle_locally(rf_bundle, "Random Forest")
            ann_saved = save_bundle_locally(ann_bundle, "ANN")
            st.success(
                f"Đã huấn luyện xong hai mô hình trên cùng {len(generated_dataset):,} mẫu: "
                f"{rf_bundle['n_train']:,} train và {rf_bundle['n_test']:,} test."
            )
            if rf_saved and ann_saved:
                st.success(f"Đã lưu mô hình vào {rf_saved} và {ann_saved}.")
            else:
                st.warning("Không ghi được một hoặc cả hai file vào thư mục models/. Bạn vẫn có thể tải file .joblib ở bên dưới.")
        except Exception as exc:
            st.error(f"Không thể huấn luyện hai mô hình: {exc}")

    # Optional import path for model artifacts generated on another computer.
    with st.expander("Nạp mô hình đã huấn luyện từ file .joblib", expanded=False):
        st.warning(
            "Chỉ nạp file .joblib do chính bạn tạo hoặc từ nguồn đáng tin cậy. "
            "Định dạng joblib/pickle có thể thực thi mã khi nạp file không đáng tin cậy."
        )
        upload_rf = st.file_uploader("Mô hình Random Forest", type=["joblib", "pkl"], key="upload_rf_bundle_v2")
        upload_ann = st.file_uploader("Mô hình ANN", type=["joblib", "pkl"], key="upload_ann_bundle_v2")
        if st.button("Nạp các mô hình đã chọn", key="load_uploaded_bundles_v2"):
            loaded_any = False
            try:
                if upload_rf is not None:
                    st.session_state[SESSION_BUNDLE_KEYS["Random Forest"]] = load_bundle_bytes(
                        upload_rf.getvalue(), "Random Forest"
                    )
                    loaded_any = True
                if upload_ann is not None:
                    st.session_state[SESSION_BUNDLE_KEYS["ANN"]] = load_bundle_bytes(
                        upload_ann.getvalue(), "ANN"
                    )
                    loaded_any = True
                if loaded_any:
                    st.success("Đã nạp mô hình vào phiên làm việc hiện tại.")
                else:
                    st.info("Bạn chưa chọn file mô hình nào.")
            except Exception as exc:
                st.error(f"Không thể nạp mô hình: {exc}")

    rf_bundle = st.session_state.get(SESSION_BUNDLE_KEYS["Random Forest"])
    ann_bundle = st.session_state.get(SESSION_BUNDLE_KEYS["ANN"])
    dataset = st.session_state.get("looplab_ai_dataset_v2")

    if rf_bundle is None or ann_bundle is None:
        st.info(
            "Chưa có đủ cả hai mô hình. Nhấn **Huấn luyện đồng thời RF + ANN** hoặc nạp hai file .joblib đã được tạo trước đó."
        )
    else:
        # Re-check expected structure before displaying metrics or running inference.
        try:
            rf_bundle = validate_model_bundle(rf_bundle, "Random Forest")
            ann_bundle = validate_model_bundle(ann_bundle, "ANN")
        except Exception as exc:
            st.error(f"Mô hình đang lưu trong phiên không tương thích: {exc}")
            st.stop()

        st.markdown("---")
        section("So sánh trực tiếp Random Forest và ANN")
        comparison_df = build_model_comparison(rf_bundle, ann_bundle)
        metrics_view = comparison_df[[
            "Đầu ra", "RF · MAE", "ANN · MAE", "MAE tốt hơn",
            "RF · RMSE", "ANN · RMSE", "RMSE tốt hơn",
            "RF · R²", "ANN · R²", "R² tốt hơn",
        ]]
        st.dataframe(
            metrics_view,
            hide_index=True,
            width="stretch",
            column_config={
                "RF · MAE": st.column_config.NumberColumn(format="%.6g"),
                "ANN · MAE": st.column_config.NumberColumn(format="%.6g"),
                "RF · RMSE": st.column_config.NumberColumn(format="%.6g"),
                "ANN · RMSE": st.column_config.NumberColumn(format="%.6g"),
                "RF · R²": st.column_config.NumberColumn(format="%.5f"),
                "ANN · R²": st.column_config.NumberColumn(format="%.5f"),
            },
        )
        st.caption(
            "MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt. "
            "Kết quả được tính trên cùng tập test vì cả hai lần gọi train_models dùng chung dataset và random_state."
        )

        chart_mae, chart_rmse, chart_r2 = st.columns(3)
        with chart_mae:
            fig_mae = go.Figure()
            fig_mae.add_bar(name="Random Forest", x=comparison_df["Đầu ra"], y=comparison_df["RF · MAE"])
            fig_mae.add_bar(name="ANN", x=comparison_df["Đầu ra"], y=comparison_df["ANN · MAE"])
            fig_mae.update_layout(title="So sánh MAE (thang log)", barmode="group", yaxis_type="log", height=330)
            fig_mae.update_xaxes(tickangle=-35, automargin=True)
            st.plotly_chart(plot_layout(fig_mae, 330), width="stretch", key="ai_compare_mae_v2")
        with chart_rmse:
            fig_rmse = go.Figure()
            fig_rmse.add_bar(name="Random Forest", x=comparison_df["Đầu ra"], y=comparison_df["RF · RMSE"])
            fig_rmse.add_bar(name="ANN", x=comparison_df["Đầu ra"], y=comparison_df["ANN · RMSE"])
            fig_rmse.update_layout(title="So sánh RMSE (thang log)", barmode="group", yaxis_type="log", height=330)
            fig_rmse.update_xaxes(tickangle=-35, automargin=True)
            st.plotly_chart(plot_layout(fig_rmse, 330), width="stretch", key="ai_compare_rmse_v2")
        with chart_r2:
            fig_r2 = go.Figure()
            fig_r2.add_bar(name="Random Forest", x=comparison_df["Đầu ra"], y=comparison_df["RF · R²"])
            fig_r2.add_bar(name="ANN", x=comparison_df["Đầu ra"], y=comparison_df["ANN · R²"])
            fig_r2.update_layout(title="So sánh R²", barmode="group", height=330)
            # Rotate category labels to avoid collisions in the narrow third column.
            fig_r2.update_xaxes(tickangle=-35, automargin=True)
            st.plotly_chart(plot_layout(fig_r2, 330), width="stretch", key="ai_compare_r2_v2")

        metric_csv = comparison_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "↓ Tải bảng so sánh RF–ANN (CSV)", metric_csv,
            "looplab_rf_ann_comparison.csv", "text/csv", key="download_rf_ann_metrics_v2"
        )

        # Save/export both full estimator bundles. The downloaded files can be
        # committed under models/ in GitHub so deployed visitors can load them.
        st.markdown("#### Lưu và xuất mô hình AI độc lập")
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            rf_bytes = serialize_bundle(rf_bundle)
            st.download_button(
                "↓ Tải mô hình Random Forest (.joblib)", rf_bytes,
                "looplab_random_forest.joblib", "application/octet-stream",
                key="download_rf_model_v2", width="stretch"
            )
        with save_col2:
            ann_bytes = serialize_bundle(ann_bundle)
            st.download_button(
                "↓ Tải mô hình ANN (.joblib)", ann_bytes,
                "looplab_ann.joblib", "application/octet-stream",
                key="download_ann_model_v2", width="stretch"
            )
        st.markdown(
            "<div class='note-box'><b>Cách lưu bền vững khi deploy:</b> tải hai file .joblib ở trên, đặt chúng trong thư mục "
            "<code>models/</code> cùng cấp với <code>app.py</code>, rồi đưa thư mục đó lên GitHub. "
            "Ứng dụng sẽ tự nạp hai mô hình từ <code>models/</code> khi khởi động. File ghi trực tiếp trên Streamlit Cloud "
            "có thể mất khi máy chủ được tạo lại, vì vậy không nên dựa riêng vào lưu trữ runtime.</div>",
            unsafe_allow_html=True,
        )

        # Current-input domain guard. Analytical simulation remains available
        # everywhere its own equations allow, but AI inference is blocked outside
        # the distribution/ranges used to build the synthetic training set.
        section("Kiểm soát miền dự đoán AI")
        domain_problems = ai_domain_violations(
            float(radius_cm), float(freq_mhz), float(wire_radius_mm), int(n_turns), material
        )
        domain_cols = st.columns(4)
        domain_descriptions = [
            ("R (cm)", "0.8 – 8.0"),
            ("f (MHz)", "5 – 50"),
            ("a (mm)", "0.25 – 1.5"),
            ("N", "1 – 4 vòng"),
        ]
        for col, (label, bounds) in zip(domain_cols, domain_descriptions):
            with col:
                st.metric(label, bounds)
        st.caption("Vật liệu hợp lệ: Đồng (Cu), Bạc (Ag), Vàng (Au), Nhôm (Al). Điều kiện vật lý bổ sung: C/λ < 0.1.")

        if domain_problems:
            st.error("Không thực hiện dự đoán AI vì thông số hiện tại nằm ngoài miền huấn luyện:")
            for problem in domain_problems:
                st.write(f"• {problem}")
            st.info(
                "Mô phỏng giải tích và các đồ thị vẫn hoạt động theo mô hình của chúng. "
                "Để dùng AI, hãy đưa thông số về miền huấn luyện nêu trên. Không nên ngoại suy từ mô hình hiện tại."
            )
        else:
            current_ai_input = {
                "radius_cm": float(radius_cm),
                "freq_mhz": float(freq_mhz),
                "wire_radius_mm": float(wire_radius_mm),
                "n_turns": int(n_turns),
                "rho_ohm_m": float(MATERIALS[material]["rho_ohm_m"]),
            }
            prediction_df = build_prediction_comparison(base, current_ai_input, rf_bundle, ann_bundle)
            section("Dự đoán của hai mô hình tại điểm làm việc hiện tại")
            st.dataframe(
                prediction_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mô hình giải tích": st.column_config.NumberColumn(format="%.7g"),
                    "Random Forest": st.column_config.NumberColumn(format="%.7g"),
                    "Sai số RF (%)": st.column_config.NumberColumn(format="%.4f"),
                    "ANN": st.column_config.NumberColumn(format="%.7g"),
                    "Sai số ANN (%)": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.download_button(
                "↓ Tải bảng dự đoán RF–ANN tại điểm hiện tại (CSV)",
                prediction_df.to_csv(index=False).encode("utf-8-sig"),
                "looplab_rf_ann_current_prediction.csv", "text/csv",
                key="download_rf_ann_prediction_v2",
            )

        if dataset is not None:
            st.markdown("#### Dataset đã dùng để huấn luyện")
            st.caption(
                f"Dataset gồm {len(dataset):,} mẫu và {len(dataset.columns)} cột. "
                "Miền sinh: R=0.8–8 cm, f=5–50 MHz, a=0.25–1.5 mm, N=1–4."
            )
            with st.expander("Xem trước 12 dòng dữ liệu"):
                st.dataframe(dataset.head(12), hide_index=True, width="stretch")
            st.download_button(
                "↓ Tải dataset huấn luyện (CSV)",
                dataset.to_csv(index=False).encode("utf-8-sig"),
                "anten_loop_ai_dataset.csv", "text/csv", key="download_ai_dataset_v2",
            )
        else:
            meta = rf_bundle.get("training_metadata", {})
            if meta:
                st.caption(
                    f"Mô hình đã nạp từ file độc lập · dataset {meta.get('dataset_rows', 'không rõ')} mẫu · "
                    f"seed {meta.get('random_state', 'không rõ')} · train/test 80/20. "
                    "Dataset gốc chưa được nạp vào phiên hiện tại."
                )

    st.markdown(
        "<div class='warning-box'><b>Giới hạn khoa học:</b> Dataset được sinh từ chính mô hình giải tích "
        "<code>antenna_core.calculate_antenna</code>. Vì vậy MAE/RMSE/R² đánh giá mức độ AI học lại mô hình giải tích, "
        "không chứng minh độ chính xác so với anten thực tế. Muốn kiểm chứng thực nghiệm cần dữ liệu VNA, buồng đo "
        "hoặc mô phỏng điện từ toàn sóng CST/HFSS/FEKO/openEMS.</div>",
        unsafe_allow_html=True,
    )

with tab_theory:
    section("Cơ sở lý thuyết và phạm vi mô hình")
    st.markdown("""
    **Các biểu thức chính đang được sử dụng**

    - Bước sóng: `λ = c / f`.
    - Diện tích vòng tròn: `A = πR²`.
    - Điện trở bức xạ vòng nhỏ: `Rr ≈ 31200 · (N·A / λ²)²`.
    - Điện cảm một vòng: `L₁ ≈ μ₀R[ln(8R/a) − 2]`; nhiều vòng dùng gần đúng `L ≈ N²L₁`.
    - Trở kháng đầu vào: `Zin ≈ (Rr + Rac) + jωL`.
    - Hiệu suất bức xạ: `ηr ≈ Rr / (Rr + Rac)`.
    - Gain gần đúng: `G ≈ D·ηr`, với `D ≈ 1.5` cho vòng nhỏ lý tưởng.
    - Điều kiện vòng nhỏ thực hành: `C/λ < 0.1`.

    **Vật liệu dây dẫn**

    | Vật liệu | Điện trở suất gần đúng (Ω·m) | Ý nghĩa trong chương trình |
    |---|---:|---|
    | Bạc (Ag) | `1.59 × 10⁻⁸` | Điện trở suất thấp nhất trong bốn vật liệu mẫu |
    | Đồng (Cu) | `1.724 × 10⁻⁸` | Vật liệu tham chiếu phổ biến |
    | Vàng (Au) | `2.44 × 10⁻⁸` | Tổn hao dẫn lớn hơn đồng trong cùng hình học |
    | Nhôm (Al) | `2.82 × 10⁻⁸` | Nhẹ hơn nhưng điện trở suất khối lớn hơn đồng |

    **Cách trình bày đề tài AI**

    1. Sinh dữ liệu cho nhiều bán kính, tần số, bán kính dây, số vòng và vật liệu.
    2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80/20.
    3. Huấn luyện Random Forest hoặc ANN để dự đoán `Rr`, `Re(Zin)`, `Im(Zin)` và `Gain`.
    4. Đánh giá bằng MAE, RMSE và R²; phân tích đầu ra AI so với mô hình giải tích.

    **Giới hạn cần nêu trong báo cáo**

    Đây là mô hình giải tích gần đúng và mô hình AI học từ dữ liệu tổng hợp. Chương trình chưa giải đầy đủ phương trình Maxwell; chưa xét điện dung khe cấp nguồn, phân bố dòng không đều chính xác, hiệu ứng lân cận giữa các vòng, tổn hao điện môi, giá đỡ, vỏ bọc, môi trường lắp đặt và nhiệt độ tăng. Hình 3D là hình học minh họa và pattern là mô hình trường xa chuẩn hóa. Không nên gọi kết quả là đo đạc hay mô phỏng full-wave.
    """)
    st.markdown("---")
    st.markdown("**Hướng triển khai trên GitHub và Streamlit Community Cloud**")
    st.code("""1. Giải nén bộ project.
2. Tạo repository GitHub mới và upload toàn bộ thư mục (app.py, antenna_core.py, ai_engine.py, requirements.txt, .streamlit/).
3. Mở share.streamlit.io và đăng nhập bằng GitHub.
4. Chọn New app → repository → branch main → Main file path: app.py.
5. Nhấn Deploy; sau khi build xong, gửi đường link cho người khác mở trên máy tính hoặc điện thoại.""", language="text")

st.markdown("<hr><div class='small-muted'>LOOPLAB · Mô hình giải tích gần đúng + ML surrogate · Dành cho mục đích học tập và nghiên cứu.</div>", unsafe_allow_html=True)
