"""
Air Quality AQI Prediction — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from datetime import date
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #f0f4f8; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a3a5c 0%, #0e2439 100%);
  }
  section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
  section[data-testid="stSidebar"] .stSlider label { color: #a0c4ff !important; }

  /* Cards */
  .metric-card {
      background: white; border-radius: 12px; padding: 20px 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 12px;
  }
  .aqi-badge {
      display: inline-block; border-radius: 50px;
      padding: 10px 28px; font-size: 2.4rem; font-weight: 800;
      letter-spacing: 2px; margin: 8px 0;
  }
  .section-header {
      font-size: 1.05rem; font-weight: 700; color: #1a3a5c;
      border-left: 4px solid #3b82d4; padding-left: 10px;
      margin: 18px 0 10px 0;
  }
  .info-text { font-size: 0.82rem; color: #6b7a8d; margin-top: 4px; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model/xgb_aqi_model.pkl")
    encoder = joblib.load("model/city_encoder.pkl")
    with open("model/model_meta.json") as f:
        meta = json.load(f)
    return model, encoder, meta

@st.cache_data
def load_dataset():
    return pd.read_csv("air_quality_dataset.csv", parse_dates=["date"])

# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_BANDS = [
    (0,   50,  "Good",             "#00e400", "#1a1a1a"),
    (51,  100, "Moderate",         "#ffff00", "#1a1a1a"),
    (101, 150, "Unhealthy for SG", "#ff7e00", "#ffffff"),
    (151, 200, "Unhealthy",        "#ff0000", "#ffffff"),
    (201, 300, "Very Unhealthy",   "#8f3f97", "#ffffff"),
    (301, 500, "Hazardous",        "#7e0023", "#ffffff"),
]

def aqi_info(aqi_val):
    for lo, hi, label, bg, fg in AQI_BANDS:
        if lo <= aqi_val <= hi:
            return label, bg, fg
    return "Hazardous", "#7e0023", "#ffffff"

AQI_HEALTH = {
    "Good":             "✅ Air quality is satisfactory. Little or no risk.",
    "Moderate":         "⚠️ Acceptable. Sensitive people may notice minor issues.",
    "Unhealthy for SG": "🟠 Sensitive groups should reduce prolonged outdoor exertion.",
    "Unhealthy":        "🔴 Everyone may begin to experience health effects.",
    "Very Unhealthy":   "🟣 Health alert — everyone may experience serious effects.",
    "Hazardous":        "☠️ Health warning: everyone should avoid outdoor activities.",
}

def _aqi_sub(c, bp):
    for lo_c, hi_c, lo_i, hi_i in bp:
        if lo_c <= c <= hi_c:
            return ((hi_i - lo_i) / (hi_c - lo_c)) * (c - lo_c) + lo_i
    return 500.0

_pm25_bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
            (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
_pm10_bp = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
            (255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def build_features(city_enc, input_date, pm25, pm10, no2, so2, co, o3, temp, humidity, wind):
    month       = input_date.month
    day_of_year = input_date.timetuple().tm_yday
    season      = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}[month]
    pm_ratio      = pm25 / (pm10 + 1e-6)
    no2_so2_ratio = no2  / (so2  + 1e-6)
    wind_temp     = wind * temp
    rh_temp       = humidity * temp
    # Log transforms
    import math
    log_pm25, log_pm10 = math.log1p(pm25), math.log1p(pm10)
    log_no2,  log_so2  = math.log1p(no2),  math.log1p(so2)
    log_co,   log_o3   = math.log1p(co),   math.log1p(o3)
    # AQI sub-indices
    sub_pm25 = _aqi_sub(pm25, _pm25_bp)
    sub_pm10 = _aqi_sub(pm10, _pm10_bp)
    sub_max  = max(sub_pm25, sub_pm10)
    wind_sq  = wind ** 2
    oxidants = no2 + o3
    return np.array([[city_enc, month, day_of_year, season,
                      pm25, pm10, no2, so2, co, o3,
                      temp, humidity, wind,
                      pm_ratio, no2_so2_ratio, wind_temp, rh_temp,
                      log_pm25, log_pm10, log_no2, log_so2, log_co, log_o3,
                      sub_pm25, sub_pm10, sub_max,
                      wind_sq, oxidants]])

# ── Gauge chart ───────────────────────────────────────────────────────────────
def aqi_gauge(aqi_val, label, bg_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        title={"text": f"<b>AQI</b><br><span style='font-size:0.85em;color:#666'>{label}</span>",
               "font": {"size": 18}},
        number={"font": {"size": 48, "color": bg_color}},
        gauge={
            "axis":  {"range": [0, 500], "tickwidth": 1, "tickcolor": "#666"},
            "bar":   {"color": bg_color, "thickness": 0.28},
            "steps": [
                {"range": [0,   50],  "color": "#00e40033"},
                {"range": [51,  100], "color": "#ffff0033"},
                {"range": [101, 150], "color": "#ff7e0033"},
                {"range": [151, 200], "color": "#ff000033"},
                {"range": [201, 300], "color": "#8f3f9733"},
                {"range": [301, 500], "color": "#7e002333"},
            ],
            "threshold": {
                "line": {"color": bg_color, "width": 4},
                "thickness": 0.8,
                "value": aqi_val,
            },
        },
    ))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ── Radar chart ───────────────────────────────────────────────────────────────
def pollutant_radar(pm25, pm10, no2, so2, co, o3):
    # Normalise against WHO / unhealthy thresholds
    norms = [pm25/55.4, pm10/154, no2/100, so2/75, co/9, o3/70]
    cats  = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃"]
    fig   = go.Figure(go.Scatterpolar(
        r=norms + [norms[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(59,130,212,0.2)",
        line=dict(color="#3b82d4", width=2),
        name="Your readings",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.5],
                                   tickvals=[0.5, 1.0, 1.5],
                                   ticktext=["50%", "100%", "150%"])),
        showlegend=False,
        height=280, margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "Pollutant Load (% of Unhealthy threshold)",
               "font": {"size": 13}, "x": 0.5},
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Check model exists
    if not os.path.exists("model/xgb_aqi_model.pkl"):
        st.error("⚠️  Model not found. Please run `python train_model.py` first.")
        st.stop()

    model, encoder, meta = load_artifacts()
    cities = meta["cities"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 18px 0 4px 0;'>
      <h1 style='color:#1a3a5c; font-size:2.2rem; margin-bottom:4px;'>
        🌿 Air Quality AQI Predictor
      </h1>
      <p style='color:#57606a; font-size:1rem; margin:0;'>
        XGBoost-powered real-time air quality index prediction
      </p>
    </div>
    <hr style='border:1px solid #e5e7eb; margin: 10px 0 20px 0;'>
    """, unsafe_allow_html=True)

    # ── Model stats banner ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    m = meta["metrics"]
    c1.metric("Model",     "XGBoost")
    c2.metric("R² Score",  f"{m['r2']:.4f}",  help="Coefficient of determination on test set")
    c3.metric("MAE",       f"{m['mae']:.2f}",  help="Mean Absolute Error (AQI points)")
    c4.metric("CV R²",     f"{meta['cv_r2_mean']:.4f} ± {meta['cv_r2_std']:.4f}",
              help="5-fold cross-validation R²")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Input Panel
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 📋 Input Parameters")
        st.markdown("---")

        # Location & Date
        st.markdown("### 📍 Location & Date")
        city        = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        input_date  = st.date_input("Date", value=date.today(),
                                    min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))

        st.markdown("---")

        # Particulate Matter
        st.markdown("### 🔴 Particulate Matter")
        pm25 = st.slider("PM2.5  (µg/m³)", 0.0, 500.0, 45.0, 0.5,
                         help="Fine particulate matter ≤ 2.5 µm diameter")
        pm10 = st.slider("PM10   (µg/m³)", 0.0, 600.0, 80.0, 1.0,
                         help="Coarse particulate matter ≤ 10 µm diameter")

        st.markdown("---")

        # Gaseous Pollutants
        st.markdown("### 🟡 Gaseous Pollutants")
        no2 = st.slider("NO₂    (µg/m³)", 0.0, 400.0, 40.0, 0.5,
                        help="Nitrogen Dioxide")
        so2 = st.slider("SO₂    (µg/m³)", 0.0, 200.0, 15.0, 0.5,
                        help="Sulfur Dioxide")
        co  = st.slider("CO     (mg/m³)", 0.0,  50.0,  1.2, 0.1,
                        help="Carbon Monoxide")
        o3  = st.slider("O₃     (µg/m³)", 0.0, 200.0, 55.0, 0.5,
                        help="Ground-level Ozone")

        st.markdown("---")

        # Meteorological
        st.markdown("### 🌤️ Meteorological")
        temperature = st.slider("Temperature (°C)",   -10.0, 50.0, 25.0, 0.1)
        humidity    = st.slider("Humidity (%)",          0.0, 100.0, 55.0, 0.5)
        wind_speed  = st.slider("Wind Speed (km/h)",     0.0,  80.0, 10.0, 0.1)

        st.markdown("---")
        predict_btn = st.button("🔮  Predict AQI", type="primary", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Explorer", "📖 AQI Reference"])

    # ── Tab 1: Prediction ─────────────────────────────────────────────────────
    with tab1:
        if predict_btn:
            city_enc  = encoder.transform([city])[0]
            X_input   = build_features(city_enc, input_date, pm25, pm10,
                                       no2, so2, co, o3,
                                       temperature, humidity, wind_speed)
            aqi_pred  = int(round(float(model.predict(X_input)[0])))
            aqi_pred  = max(0, min(500, aqi_pred))
            label, bg_color, fg_color = aqi_info(aqi_pred)
            health_msg = AQI_HEALTH[label]

            # Result layout
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                  <div style='font-size:1.1rem; font-weight:600; color:#57606a; margin-bottom:6px;'>
                    Predicted AQI for <b>{city}</b> on <b>{input_date}</b>
                  </div>
                  <div class='aqi-badge' style='background:{bg_color}; color:{fg_color};'>
                    {aqi_pred}
                  </div>
                  <div style='font-size:1.3rem; font-weight:700; color:{bg_color if bg_color not in ["#ffff00","#00e400"] else "#1a1a1a"};
                              margin:6px 0;'>
                    {label}
                  </div>
                  <div style='font-size:0.95rem; color:#374151; margin-top:10px; padding:10px;
                              background:#f9fafb; border-radius:8px;'>
                    {health_msg}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Input summary table
                st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
                summary_df = pd.DataFrame({
                    "Parameter": ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃",
                                  "Temperature", "Humidity", "Wind Speed"],
                    "Value": [f"{pm25} µg/m³", f"{pm10} µg/m³", f"{no2} µg/m³",
                              f"{so2} µg/m³",  f"{co} mg/m³",   f"{o3} µg/m³",
                              f"{temperature} °C", f"{humidity} %", f"{wind_speed} km/h"],
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            with col_right:
                st.plotly_chart(aqi_gauge(aqi_pred, label, bg_color),
                                use_container_width=True)
                st.plotly_chart(pollutant_radar(pm25, pm10, no2, so2, co, o3),
                                use_container_width=True)

            # AQI breakdown bar
            st.markdown("<div class='section-header'>AQI Scale Position</div>",
                        unsafe_allow_html=True)
            scale_fig = go.Figure()
            for lo, hi, lbl, bg, _ in AQI_BANDS:
                scale_fig.add_trace(go.Bar(
                    x=[hi - lo], base=[lo], y=["AQI Scale"],
                    orientation="h", marker_color=bg,
                    name=lbl, hovertemplate=f"{lbl}: {lo}–{hi}<extra></extra>",
                ))
            scale_fig.add_vline(x=aqi_pred, line_color="#1a3a5c", line_width=3,
                                annotation_text=f"  {aqi_pred}", annotation_position="top")
            scale_fig.update_layout(
                barmode="stack", showlegend=True, height=140,
                margin=dict(l=10, r=10, t=10, b=30),
                legend=dict(orientation="h", y=-0.8, font=dict(size=11)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 500], title="AQI"),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(scale_fig, use_container_width=True)

        else:
            # Placeholder when no prediction yet
            st.markdown("""
            <div style='text-align:center; padding: 60px 20px; background:white;
                        border-radius:16px; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
              <div style='font-size:4rem; margin-bottom:16px;'>🌿</div>
              <div style='font-size:1.4rem; font-weight:700; color:#1a3a5c;'>
                Ready to Predict Air Quality
              </div>
              <div style='color:#57606a; margin-top:10px; font-size:1rem;'>
                Adjust the sliders in the sidebar and click
                <b>Predict AQI</b> to get an instant AQI prediction
                powered by XGBoost.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Data Explorer ──────────────────────────────────────────────────
    with tab2:
        if not os.path.exists("air_quality_dataset.csv"):
            st.info("Dataset file not found. Run `python generate_dataset.py` first.")
        else:
            df = load_dataset()
            st.markdown(f"**Dataset:** {len(df):,} rows · {df.shape[1]} features")

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            sel_cities = fc1.multiselect("Filter cities",
                                         sorted(df["city"].unique()),
                                         default=["Delhi", "Mumbai", "Bangalore"])
            date_range = fc2.date_input(
                "Date range",
                [df["date"].min().date(), df["date"].max().date()],
                key="dr")
            aqi_range  = fc3.slider("AQI range", 0, 500, (0, 500), key="aqir")

            fdf = df.copy()
            if sel_cities:
                fdf = fdf[fdf["city"].isin(sel_cities)]
            if len(date_range) == 2:
                fdf = fdf[(fdf["date"].dt.date >= date_range[0]) &
                          (fdf["date"].dt.date <= date_range[1])]
            fdf = fdf[(fdf["aqi"] >= aqi_range[0]) & (fdf["aqi"] <= aqi_range[1])]

            st.markdown(f"*Showing {len(fdf):,} rows after filters*")

            # Charts row
            ch1, ch2 = st.columns(2)
            with ch1:
                avg_aqi = fdf.groupby("city")["aqi"].mean().sort_values(ascending=True)
                fig_bar = px.bar(avg_aqi, orientation="h",
                                 labels={"value": "Avg AQI", "index": "City"},
                                 title="Average AQI by City",
                                 color=avg_aqi.values,
                                 color_continuous_scale=["#00e400","#ffff00","#ff7e00","#ff0000","#7e0023"])
                fig_bar.update_layout(height=340, showlegend=False,
                                      coloraxis_showscale=False,
                                      paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with ch2:
                monthly = fdf.copy()
                monthly["month"] = monthly["date"].dt.month
                monthly_avg = monthly.groupby("month")["aqi"].mean().reset_index()
                monthly_avg["month_name"] = monthly_avg["month"].map(
                    {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
                fig_line = px.line(monthly_avg, x="month_name", y="aqi",
                                   markers=True, title="Monthly AQI Trend",
                                   labels={"aqi": "Avg AQI", "month_name": "Month"})
                fig_line.update_traces(line_color="#3b82d4")
                fig_line.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_line, use_container_width=True)

            # Scatter
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_sc = px.scatter(fdf.sample(min(500, len(fdf)), random_state=1),
                                    x="pm25", y="aqi", color="city",
                                    opacity=0.65, title="PM2.5 vs AQI",
                                    labels={"pm25":"PM2.5 (µg/m³)", "aqi":"AQI"})
                fig_sc.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc, use_container_width=True)

            with sc2:
                fig_sc2 = px.scatter(fdf.sample(min(500, len(fdf)), random_state=2),
                                     x="wind_speed", y="aqi", color="city",
                                     opacity=0.65, title="Wind Speed vs AQI",
                                     labels={"wind_speed":"Wind Speed (km/h)", "aqi":"AQI"})
                fig_sc2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc2, use_container_width=True)

            # Correlation heatmap
            st.markdown("<div class='section-header'>Correlation Heatmap</div>",
                        unsafe_allow_html=True)
            num_cols = ["pm25","pm10","no2","so2","co","o3",
                        "temperature","humidity","wind_speed","aqi"]
            corr = fdf[num_cols].corr().round(2)
            fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                                 color_continuous_scale="RdBu_r",
                                 title="Feature Correlation Matrix")
            fig_heat.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Raw data
            with st.expander("📄 Raw Data Table"):
                st.dataframe(fdf.head(200).reset_index(drop=True),
                             use_container_width=True)

    # ── Tab 3: AQI Reference ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### AQI Scale — US EPA Standard")
        ref_data = [
            ("0–50",   "Good",                "#00e400", "#1a1a1a",
             "Air quality is satisfactory; little or no risk."),
            ("51–100", "Moderate",             "#ffff00", "#1a1a1a",
             "Acceptable; some pollutants may be a concern for very sensitive people."),
            ("101–150","Unhealthy for Sensitive Groups", "#ff7e00", "#ffffff",
             "General public is not likely to be affected; sensitive groups at risk."),
            ("151–200","Unhealthy",            "#ff0000", "#ffffff",
             "Everyone may begin to experience health effects."),
            ("201–300","Very Unhealthy",       "#8f3f97", "#ffffff",
             "Health alert — everyone may experience serious effects."),
            ("301–500","Hazardous",            "#7e0023", "#ffffff",
             "Emergency conditions — entire population likely affected."),
        ]
        for rng, lbl, bg, fg, desc in ref_data:
            st.markdown(f"""
            <div style='display:flex; align-items:center; background:{bg};
                        color:{fg}; border-radius:10px; padding:12px 18px;
                        margin-bottom:8px; gap:16px;'>
              <div style='font-size:1.3rem; font-weight:800; min-width:70px;'>{rng}</div>
              <div style='font-size:1.1rem; font-weight:700; min-width:200px;'>{lbl}</div>
              <div style='font-size:0.92rem; opacity:0.92;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Pollutant Reference Guide")
        poll_ref = {
            "Pollutant":     ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃"],
            "Unit":          ["µg/m³", "µg/m³", "µg/m³", "µg/m³", "mg/m³", "µg/m³"],
            "WHO Guideline": ["15 (24h)", "45 (24h)", "25 (24h)", "40 (24h)", "4 (24h)", "100 (8h)"],
            "Unhealthy Level":["55.5+",   "155+",     "101+",     "76+",      "9.5+",    "71+"],
            "Primary Source": [
                "Combustion, industry, vehicles",
                "Dust, pollen, construction",
                "Vehicle exhaust, power plants",
                "Burning coal & oil",
                "Incomplete combustion",
                "Sunlight + NOx + VOCs",
            ],
        }
        st.dataframe(pd.DataFrame(poll_ref), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Feature Importance (Model)")
        if "feature_importance" in meta:
            fi = pd.DataFrame(
                list(meta["feature_importance"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Blues",
                            title="XGBoost Feature Importances")
            fig_fi.update_layout(height=420, showlegend=False,
                                 coloraxis_showscale=False,
                                 paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fi, use_container_width=True)


if __name__ == "__main__":
    main()
