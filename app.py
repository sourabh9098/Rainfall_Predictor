import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RainCast — Rainfall Predictor",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Mulish:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Mulish', sans-serif; }

.stApp {
    background: #eaf4fb;
    min-height: 100vh;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 15% 10%, rgba(56,139,202,0.13) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 90%, rgba(30,80,160,0.10) 0%, transparent 60%);
    z-index: 0;
}
.block-container {
    position: relative;
    z-index: 1;
    max-width: 800px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0c2461 0%, #1e3799 45%, #0a3d62 100%);
    border-radius: 28px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(12,36,97,0.30);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -70px; right: -70px;
    width: 250px; height: 250px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50px; left: -50px;
    width: 190px; height: 190px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #a8d8f0;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.hero-title span { color: #74b9ff; }
.hero-sub {
    color: rgba(168,216,240,0.75);
    font-size: 0.92rem;
    font-weight: 300;
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-models {
    display: flex;
    justify-content: center;
    gap: 0.7rem;
    margin-top: 1.3rem;
    flex-wrap: wrap;
}
.model-tag {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(200,230,255,0.8);
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    text-transform: uppercase;
}
.model-tag.best {
    background: rgba(116,185,255,0.18);
    border-color: rgba(116,185,255,0.45);
    color: #74b9ff;
}

/* ── card ── */
.card {
    background: #ffffff;
    border: 1px solid #d6eaf8;
    border-radius: 20px;
    padding: 1.8rem 2rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(12,36,97,0.06);
    transition: box-shadow 0.3s;
}
.card:hover { box-shadow: 0 8px 32px rgba(12,36,97,0.11); }

.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1e3799;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, #a8d8f0, transparent);
}

/* widgets */
label[data-testid="stWidgetLabel"] p {
    color: #1a2a4a !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div {
    background: #f0f8ff !important;
    border: 1.5px solid #a8d8f0 !important;
    border-radius: 10px !important;
    color: #1a2a4a !important;
}
div[data-testid="stSlider"] { padding: 0.2rem 0; }
div[data-testid="stNumberInput"] input {
    background: #f0f8ff !important;
    border: 1.5px solid #a8d8f0 !important;
    border-radius: 10px !important;
    color: #1a2a4a !important;
}

/* ── predict button ── */
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #0c2461, #1e3799);
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2rem;
    cursor: pointer;
    box-shadow: 0 6px 24px rgba(30,55,153,0.35);
    transition: all 0.3s ease;
    margin-top: 0.8rem;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 36px rgba(30,55,153,0.45);
}

/* ── result ── */
.result-wrap { margin-top: 2rem; animation: riseUp 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
@keyframes riseUp {
    from { opacity:0; transform: translateY(30px) scale(0.95); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

.result-rain {
    background: linear-gradient(135deg, #0c2461, #1e3799);
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(12,36,97,0.35);
}
.result-norain {
    background: linear-gradient(135deg, #1a5276, #2e86c1);
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(26,82,118,0.30);
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-status {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.result-msg {
    color: rgba(255,255,255,0.65);
    font-size: 0.88rem;
    font-weight: 300;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
}

/* confidence bar */
.conf-wrap { margin-top: 1.6rem; text-align: left; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.conf-track {
    background: rgba(255,255,255,0.12);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.conf-fill-rain {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #74b9ff, #0984e3);
    box-shadow: 0 0 12px rgba(116,185,255,0.5);
}
.conf-fill-sun {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #ffeaa7, #fdcb6e);
    box-shadow: 0 0 12px rgba(253,203,110,0.5);
}

/* metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0.7rem;
    margin-top: 1.4rem;
}
.metric-tile {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 0.85rem 0.5rem;
    text-align: center;
}
.mt-val { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #fff; }
.mt-lbl { font-size: 0.58rem; color: rgba(255,255,255,0.45); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.2rem; }

/* tips */
.tips-card {
    background: #eaf4fb;
    border: 1px solid #a8d8f0;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.tips-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1e3799;
    margin-bottom: 0.6rem;
}
.tips-text { font-size: 0.83rem; color: #1a2a4a; line-height: 1.65; }

.footer {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.72rem;
    margin-top: 2.5rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Naive_model.pkl")

try:
    model = load_model()
    loaded = True
except Exception as e:
    loaded = False
    load_err = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Gaussian Naive Bayes · Weather Intelligence</div>
  <div class="hero-title">Will It <span>Rain</span> Today?</div>
  <div class="hero-sub">Enter today's weather conditions and get an instant AI-powered rainfall prediction.</div>
  <div class="hero-models">
    <span class="model-tag">KNN — Tested</span>
    <span class="model-tag">Decision Tree — Tested</span>
    <span class="model-tag best">✦ Naive Bayes — Best Model</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error(f"Could not load model. Place `Naive_model.pkl` in the same folder.\n\n{load_err}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Atmospheric Conditions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">🌡️ Atmospheric Conditions</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pressure = st.slider("Pressure (hPa)", 990.0, 1040.0, 1013.0, 0.1,
                         help="Atmospheric pressure in hectopascals")
with col2:
    dewpoint = st.slider("Dew Point (°C)", 0.0, 30.0, 15.0, 0.1,
                         help="Temperature at which air becomes saturated with moisture")

col3, col4 = st.columns(2)
with col3:
    humidity = st.slider("Humidity (%)", 0, 100, 72,
                         help="Relative humidity percentage")
with col4:
    cloud = st.slider("Cloud Cover (%)", 0, 100, 50,
                      help="Percentage of sky covered by clouds")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Sun & Wind
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">💨 Sunshine & Wind</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    sunshine = st.slider("Sunshine (hours)", 0.0, 14.0, 6.0, 0.1,
                         help="Number of sunshine hours during the day")
with col6:
    windspeed = st.slider("Wind Speed (km/h)", 0.0, 60.0, 15.0, 0.5,
                          help="Wind speed in kilometres per hour")

winddirection = st.slider("Wind Direction (°)", 0.0, 360.0, 180.0, 1.0,
                           help="Wind direction in degrees — 0/360 = North, 90 = East, 180 = South, 270 = West")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔍  Predict Rainfall"):
    with st.spinner("Analysing weather conditions..."):
        try:
            # features order: pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed
            X     = np.array([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]])
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            conf  = round(float(max(proba)) * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-rain">
                    <div class="result-icon">🌧️</div>
                    <div class="result-status">Rain Expected</div>
                    <div class="result-msg">The atmospheric conditions indicate rainfall is likely today. Consider carrying an umbrella and planning accordingly.</div>
                    <div class="conf-wrap">
                      <div class="conf-label"><span>Prediction Confidence</span><span>{conf}%</span></div>
                      <div class="conf-track"><div class="conf-fill-rain" style="width:{conf}%;"></div></div>
                    </div>
                    <div class="metrics-grid">
                      <div class="metric-tile"><div class="mt-val">{pressure}</div><div class="mt-lbl">Pressure hPa</div></div>
                      <div class="metric-tile"><div class="mt-val">{humidity}%</div><div class="mt-lbl">Humidity</div></div>
                      <div class="metric-tile"><div class="mt-val">{cloud}%</div><div class="mt-lbl">Cloud Cover</div></div>
                      <div class="metric-tile"><div class="mt-val">{sunshine}h</div><div class="mt-lbl">Sunshine</div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tips-card">
                  <div class="tips-title">Weather Advisory</div>
                  <div class="tips-text">High humidity combined with strong cloud cover are the main indicators driving this prediction. Low sunshine hours and a dropping dew point further support the likelihood of rain. Plan outdoor activities for the morning if possible.</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-norain">
                    <div class="result-icon">☀️</div>
                    <div class="result-status">No Rain Expected</div>
                    <div class="result-msg">Current weather conditions suggest it is unlikely to rain today. A relatively clear day is ahead.</div>
                    <div class="conf-wrap">
                      <div class="conf-label"><span>Prediction Confidence</span><span>{conf}%</span></div>
                      <div class="conf-track"><div class="conf-fill-sun" style="width:{conf}%;"></div></div>
                    </div>
                    <div class="metrics-grid">
                      <div class="metric-tile"><div class="mt-val">{pressure}</div><div class="mt-lbl">Pressure hPa</div></div>
                      <div class="metric-tile"><div class="mt-val">{humidity}%</div><div class="mt-lbl">Humidity</div></div>
                      <div class="metric-tile"><div class="mt-val">{cloud}%</div><div class="mt-lbl">Cloud Cover</div></div>
                      <div class="metric-tile"><div class="mt-val">{sunshine}h</div><div class="mt-lbl">Sunshine</div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tips-card">
                  <div class="tips-title">Weather Summary</div>
                  <div class="tips-text">Good sunshine hours and lower cloud cover are the primary signals for a dry day. Atmospheric pressure is within a stable range. Conditions look favourable for outdoor plans, though always check updated forecasts for rapidly changing weather.</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("""
<div class="footer">
  RainCast &nbsp;·&nbsp; Gaussian Naive Bayes Model &nbsp;·&nbsp; Built with Streamlit<br>
  Dataset: 366 daily weather records · 7 features after multicollinearity removal
</div>
""", unsafe_allow_html=True)