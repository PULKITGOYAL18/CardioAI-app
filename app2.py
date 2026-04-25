import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
from datetime import date, datetime
# ─── PAGE CONFIG ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="CardioAI - Heart Disease Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SESSION STATE INIT ───────────────────────────────────────────────────── #
if "booking_open" not in st.session_state:
    st.session_state.booking_open = {}

if "booking_confirmed" not in st.session_state:
    st.session_state.booking_confirmed = {}

if "lime_done" not in st.session_state:
    st.session_state.lime_done = False

if "lime_fig" not in st.session_state:
    st.session_state.lime_fig = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = {}

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F9FC;
    color: #1B2A4A;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D47A1 0%, #1565C0 60%, #1976D2 100%);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E3F2FD !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 2px 16px rgba(13,71,161,0.08);
    margin-bottom: 20px;
    border: 1px solid #E8EEF7;
}
.card-blue {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    color: white !important;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.card-blue * { color: white !important; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #0D47A1;
    margin-bottom: 4px;
}
.section-sub { font-size: 14px; color: #607D8B; margin-bottom: 24px; }

.metric-box {
    background: #EFF4FF;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #DBEAFE;
    margin-bottom: 10px;
}
.metric-box .val { font-size: 28px; font-weight: 700; color: #1565C0; }
.metric-box .lbl { font-size: 12px; color: #607D8B; margin-top: 2px; }

.doc-card {
    background: #fff;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #E8EEF7;
    box-shadow: 0 1px 8px rgba(13,71,161,0.06);
    margin-bottom: 10px;
}
.booking-form {
    background: #F0F7FF;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #BBDEFB;
    margin-bottom: 16px;
}
.fancy-hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #1565C0 0%, #E3F2FD 100%);
    margin: 32px 0;
    border-radius: 2px;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: #CFD8DC !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    color: white; border: none; border-radius: 10px;
    padding: 10px 28px; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0D47A1, #1565C0);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(21,101,192,0.35);
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────── #
@st.cache_resource
def load_model():
    mdl = joblib.load("heart_model.pkl")
    mp  = joblib.load("label_mapping.pkl")
    rev = {v: k for k, v in mp.items()}
    return mdl, mp, rev

try:
    model, mapping, reverse_mapping = load_model()
    MODEL_LOADED = True
    MODEL_ERROR  = ""
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)

# ─── CONSTANTS ────────────────────────────────────────────────────────────── #
FEATURE_NAMES = [
    "Age", "Gender", "Chest Pain", "Shortness of Breath",
    "Chest Tightness", "Palpitations", "Irregular Heartbeat",
    "Dizziness", "Fatigue", "Leg Swelling",
    "High BP", "High Cholesterol", "Exercise Pain",
]

CITIES = ["Delhi", "Mumbai", "Bangalore", "Haridwar", "Chennai", "Kolkata", "Pune", "Hyderabad"]

DOCTORS = [
    {"name": "Dr. Ravikant Sharma", "city": "Delhi",     "specialist": "Senior Cardiologist",         "fees": "Rs.800",  "exp": "18 yrs", "hospital": "Apollo Hospital"},
    {"name": "Dr. Priya Malhotra",  "city": "Delhi",     "specialist": "Interventional Cardiologist", "fees": "Rs.1200", "exp": "14 yrs", "hospital": "Fortis Heart Institute"},
    {"name": "Dr. Avinash Mehta",   "city": "Mumbai",    "specialist": "Cardiologist",                "fees": "Rs.1000", "exp": "20 yrs", "hospital": "Lilavati Hospital"},
    {"name": "Dr. Sneha Patil",     "city": "Mumbai",    "specialist": "Cardiac Electrophysiologist", "fees": "Rs.1500", "exp": "12 yrs", "hospital": "KEM Hospital"},
    {"name": "Dr. Punit Sadana",    "city": "Bangalore", "specialist": "Cardiologist",                "fees": "Rs.900",  "exp": "16 yrs", "hospital": "Manipal Hospital"},
    {"name": "Dr. Ananya Rao",      "city": "Bangalore", "specialist": "Preventive Cardiologist",     "fees": "Rs.700",  "exp": "10 yrs", "hospital": "Narayana Health"},
    {"name": "Dr. Tarun Gupta",     "city": "Haridwar",  "specialist": "Cardiologist",                "fees": "Rs.500",  "exp": "22 yrs", "hospital": "Patanjali Hospital"},
    {"name": "Dr. Sanjay Kapoor",   "city": "Chennai",   "specialist": "Cardiologist",                "fees": "Rs.950",  "exp": "15 yrs", "hospital": "MGM Healthcare"},
    {"name": "Dr. Lakshmi Nair",    "city": "Kolkata",   "specialist": "Cardiologist",                "fees": "Rs.850",  "exp": "13 yrs", "hospital": "AMRI Hospital"},
    {"name": "Dr. Amit Desai",      "city": "Pune",      "specialist": "Cardiologist",                "fees": "Rs.750",  "exp": "11 yrs", "hospital": "Jehangir Hospital"},
    {"name": "Dr. Rekha Reddy",     "city": "Hyderabad", "specialist": "Cardiologist",                "fees": "Rs.1100", "exp": "17 yrs", "hospital": "KIMS Hospital"},
]

DISEASE_INFO = {
    "Heart Attack": {
        "emoji": "🚨",
        "desc": "A heart attack (myocardial infarction) occurs when blood flow to a part of the heart is blocked for long enough that part of the heart muscle is damaged or dies.",
        "causes": ["Coronary artery disease", "Blood clot formation", "Severe artery spasm", "Plaque build-up in arteries"],
        "precautions": ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest and avoid exertion", "Do NOT eat or drink anything"],
        "diet": ["Low-sodium diet", "Avoid saturated fats", "Leafy greens, berries, whole grains", "Omega-3 rich fish like salmon"],
        "alert_type": "danger",
    },
    "Angina": {
        "emoji": "⚠️",
        "desc": "Angina is chest pain or discomfort caused by reduced blood flow to the heart muscle. It is a symptom of coronary artery disease.",
        "causes": ["Narrowed coronary arteries", "Physical exertion or stress", "Cold weather exposure", "Heavy meals"],
        "precautions": ["Rest when pain occurs", "Use prescribed nitrates", "Avoid strenuous activity", "Regular cardiology follow-ups"],
        "diet": ["Mediterranean diet", "Reduce red meat intake", "More fruits and vegetables", "Limit alcohol and caffeine"],
        "alert_type": "warning",
    },
    "Arrhythmia": {
        "emoji": "💓",
        "desc": "Arrhythmia refers to an irregular heartbeat that is either too fast, too slow, or with an irregular pattern that disrupts normal heart function.",
        "causes": ["Electrolyte imbalance", "Thyroid disorders", "Structural heart disease", "Excessive caffeine or stress"],
        "precautions": ["Monitor heart rate regularly", "Avoid stimulants", "Wear a Holter monitor if prescribed", "Stay well hydrated"],
        "diet": ["Potassium-rich foods (bananas, spinach)", "Avoid excessive caffeine", "Magnesium-rich foods", "Low-sugar diet"],
        "alert_type": "info",
    },
    "No Disease": {
        "emoji": "✅",
        "desc": "No significant cardiac condition detected based on the provided symptoms. Maintain a healthy lifestyle to stay heart-healthy.",
        "causes": ["N/A"],
        "precautions": ["Regular annual check-ups", "Exercise 30 minutes per day", "Manage stress with yoga or meditation", "Get 7 to 8 hours of sleep"],
        "diet": ["Balanced diet with all food groups", "Plenty of water daily", "Limit processed food", "Seasonal fruits and vegetables"],
        "alert_type": "success",
    },
}

# ─── REAL CONFUSION MATRIX ────────────────────────────────────────────────── #
REAL_CM = np.array([
    [382,   0,   0,   0],
    [  0, 369,   0,  13],
    [  0,  10, 407,   6],
    [  0,  15,   0, 398],
])

# ─── REAL CLASSIFICATION REPORT ───────────────────────────────────────────── #
REAL_REPORT = {
    "No Disease":   {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 382},
    "Heart Attack": {"precision": 0.94, "recall": 0.97, "f1": 0.95, "support": 382},
    "Arrhythmia":   {"precision": 1.00, "recall": 0.96, "f1": 0.98, "support": 423},
    "Angina":       {"precision": 0.95, "recall": 0.96, "f1": 0.96, "support": 413},
    "accuracy":     {"precision": None, "recall": None, "f1": 0.97, "support": 1600},
    "macro avg":    {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 1600},
    "weighted avg": {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 1600},
}
CM_LABELS = ["No Disease", "Heart Attack", "Arrhythmia", "Angina"]

# ─── HELPERS ──────────────────────────────────────────────────────────────── #
def convert_input(val):
    if val == "Yes":
        return 1
    if val == "No":
        return 0
    return 0.5


def yn_select(label, key):
    return st.selectbox(label, ["No", "Yes", "Don't Know"], key=key)


def alert_box(msg, kind="info"):
    palette = {
        "danger":  ("#FFEBEE", "#C62828", "#EF9A9A", "🚨"),
        "warning": ("#FFF8E1", "#E65100", "#FFCC80", "⚠️"),
        "info":    ("#E3F2FD", "#0D47A1", "#90CAF9", "ℹ️"),
        "success": ("#E8F5E9", "#1B5E20", "#A5D6A7", "✅"),
    }
    bg, fc, bc, ico = palette[kind]
    st.markdown(f"""
    <div style="background:{bg};border:1.5px solid {bc};border-radius:12px;
                padding:14px 20px;color:{fc};font-weight:500;margin:10px 0">
        {ico} {msg}
    </div>""", unsafe_allow_html=True)


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def safe_str(s):
    return (str(s)
        .replace("\u2013", "-").replace("\u2014", "--")
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2022", "*").replace("\u00b7", "*")
        .replace("\u20b9", "Rs.")
        .encode("latin-1", errors="replace")
        .decode("latin-1"))


# ─── SIDEBAR ──────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px">
        <div style="font-size:52px">🫀</div>
        <div style="font-family:'DM Serif Display',serif;font-size:22px;font-weight:700;color:#fff">CardioAI</div>
        <div style="font-size:12px;color:#90CAF9;margin-top:4px">Heart Disease Detection System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2)'>", unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠 Home", "🔮 Prediction", "📊 Model Insights", "ℹ️ About"])

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2)'>", unsafe_allow_html=True)
    if MODEL_LOADED:
        st.markdown("""
        <div style="font-size:12px;color:#90CAF9;text-align:center">
            ✅ Model loaded successfully<br>
            <span style="color:#A5D6A7">Voting Classifier (RF + SVM)</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-size:12px;color:#EF9A9A;text-align:center">
            Model not loaded<br><small>{MODEL_ERROR}</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;color:#90CAF9;text-align:center;margin-top:30px">
        2nd Year Project 2025-26<br>
        Atul | Raza | Pulkit | Saurabh
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D47A1,#1976D2);border-radius:20px;
                padding:40px;margin-bottom:28px;color:white;">
        <div style="font-family:'DM Serif Display',serif;font-size:38px;font-weight:700;line-height:1.2">
            AI-Powered Heart<br>Disease Detection
        </div>
        <div style="font-size:16px;opacity:0.85;margin-top:12px;max-width:480px">
            Enter your symptoms and get an instant AI-based prediction, LIME explainability,
            doctor recommendations, and a downloadable health report.
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (ico, val, lbl) in zip([c1, c2, c3, c4], [
        ("🏥", "4 Diseases", "Detection classes"),
        ("📊", "13 Features", "Symptom inputs"),
        ("🧠", "LIME", "Explainability"),
        ("📄", "PDF Report", "Downloadable"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:28px">{ico}</div>
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown("""
        <div class="card">
            <div style="font-size:20px;font-weight:700;color:#0D47A1;margin-bottom:12px">🔍 Detectable Conditions</div>
            <div style="line-height:2.2">
                🚨 <b>Heart Attack</b> - Myocardial Infarction<br>
                ⚠️ <b>Angina</b> - Chest pain from reduced blood flow<br>
                💓 <b>Arrhythmia</b> - Irregular heart rhythm<br>
                ✅ <b>No Disease</b> - Clinically normal
            </div>
        </div>""", unsafe_allow_html=True)
    with cr:
        st.markdown("""
        <div class="card">
            <div style="font-size:20px;font-weight:700;color:#0D47A1;margin-bottom:12px">🤖 ML Pipeline</div>
            <div style="line-height:2.2">
                🌲 <b>Random Forest</b> - Tuned with GridSearchCV<br>
                🧮 <b>SVM</b> - Optimized kernel classifier<br>
                🗳️ <b>Voting Classifier</b> - Soft voting ensemble<br>
                🔬 <b>LIME</b> - Local explanation engine
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-left:4px solid #1565C0">
        <div style="font-size:20px;font-weight:700;color:#0D47A1;margin-bottom:12px">📂 Dataset Overview</div>
        <div style="display:flex;gap:40px;flex-wrap:wrap;font-size:14px">
            <div><b>Source:</b> Kaggle Diseases Dataset</div>
            <div><b>Size:</b> ~1.5 Lakh rows</div>
            <div><b>Raw Features:</b> 150+</div>
            <div><b>Selected Features:</b> 13</div>
            <div><b>Method:</b> Feature importance + domain knowledge</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**👈 Use the sidebar to navigate to Prediction**")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    if not MODEL_LOADED:
        st.error(
            f"Model files not found. Place heart_model.pkl and label_mapping.pkl "
            f"in the app folder.\n\nError: {MODEL_ERROR}"
        )
        st.stop()

    section_header("🔮 Heart Disease Prediction", "Fill in patient details and symptoms below")

    # ── Patient info ───────────────────────────────────────────────────────── #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Patient Information")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        patient_name = st.text_input("Patient Name", placeholder="e.g. Rahul Sharma", key="pname")
    with p2:
        age = st.slider("Age", 20, 90, 40, key="age_s")
    with p3:
        sex_label = st.selectbox("Gender", ["Male", "Female"], key="sex")
        sex = 1 if sex_label == "Male" else 0
    with p4:
        city = st.selectbox("City", CITIES, key="city")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Symptoms ───────────────────────────────────────────────────────────── #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🩺 Symptom Assessment")
    st.caption("Select Yes / No / Don't Know for each symptom")

    SYM = [
        ("Chest Pain 💔", "cp"),         ("Shortness of Breath 🌬️", "sob"), ("Chest Tightness 🫀", "ct"),
        ("Palpitations 💓", "pal"),       ("Irregular Heartbeat ⚡", "irr"), ("Dizziness 😵", "diz"),
        ("Fatigue 😴", "fat"),            ("Leg Swelling 🦵", "sw"),         ("High Blood Pressure 🩸", "hbp"),
        ("High Cholesterol 🔬", "hcol"),  ("Pain During Exercise 🏃", "ep"),
    ]
    raw_vals = {}
    for row_items in [SYM[i:i+3] for i in range(0, len(SYM), 3)]:
        cols = st.columns(3)
        for col, (label, key) in zip(cols, row_items):
            with col:
                raw_vals[key] = yn_select(label, key)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Build input vector ─────────────────────────────────────────────────── #
    SKEYS = ["cp", "sob", "ct", "pal", "irr", "diz", "fat", "sw", "hbp", "hcol", "ep"]
    input_data  = np.array([[age, sex] + [convert_input(raw_vals[k]) for k in SKEYS]])
    has_unknown = 0.5 in input_data[0]

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Run Prediction", use_container_width=False):
        with st.spinner("Analyzing symptoms with AI..."):
            prediction    = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
        disease    = reverse_mapping[prediction]
        confidence = float(np.max(probabilities)) * 100
        info       = DISEASE_INFO.get(disease, DISEASE_INFO["No Disease"])

        st.session_state.prediction_done = True
        st.session_state.lime_done = False
        st.session_state.lime_fig  = None
        st.session_state.prediction_data = dict(
            prediction=prediction, probabilities=probabilities,
            disease=disease, confidence=confidence, info=info,
            patient_name=patient_name, age=age, sex_label=sex_label,
            city=city, input_data=input_data, raw_vals=raw_vals,
            has_unknown=has_unknown,
        )

    # ── Render results ─────────────────────────────────────────────────────── #
    if st.session_state.prediction_done and st.session_state.prediction_data:
        D = st.session_state.prediction_data

        prediction    = D["prediction"]
        probabilities = D["probabilities"]
        disease       = D["disease"]
        confidence    = D["confidence"]
        info          = D["info"]
        patient_name  = D["patient_name"]
        age           = D["age"]
        sex_label     = D["sex_label"]
        city          = D["city"]
        input_data    = D["input_data"]
        raw_vals      = D["raw_vals"]
        has_unknown   = D["has_unknown"]
        city_doctors  = [d for d in DOCTORS if d["city"] == city]

        # Symptom keys needed later for PDF
        SKEYS = ["cp", "sob", "ct", "pal", "irr", "diz", "fat", "sw", "hbp", "hcol", "ep"]

        st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
        section_header("🩺 Prediction Result")

        r1, r2 = st.columns([1.2, 1])
        with r1:
            st.markdown(f"""
            <div class="card" style="border-left:5px solid #1565C0">
                <div style="font-size:12px;color:#607D8B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
                    Detected Condition
                </div>
                <div style="font-size:36px;font-weight:800;color:#0D47A1">
                    {info['emoji']} {disease}
                </div>
                <div style="font-size:15px;color:#455A64;margin-top:14px">
                    AI Confidence: <b style="color:#1565C0">{confidence:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)
            if has_unknown:
                alert_box("Some symptoms marked Don't Know. Prediction may be less certain.", "warning")

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                number={"suffix": "%", "font": {"size": 28, "color": "#0D47A1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#CFD8DC"},
                    "bar": {"color": "#1565C0", "thickness": 0.25},
                    "bgcolor": "#EFF4FF", "bordercolor": "#DBEAFE",
                    "steps": [
                        {"range": [0, 40],   "color": "#E3F2FD"},
                        {"range": [40, 70],  "color": "#BBDEFB"},
                        {"range": [70, 100], "color": "#90CAF9"},
                    ],
                    "threshold": {
                        "line": {"color": "#C62828", "width": 3},
                        "thickness": 0.75,
                        "value": confidence,
                    },
                },
                title={"text": "Confidence Score", "font": {"size": 13, "color": "#607D8B"}},
            ))
            fig_g.update_layout(
                height=230, margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="white", font={"family": "DM Sans"},
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # Probability bar chart
        try:
            n_cls     = len(probabilities)
            cls_names = [reverse_mapping[i] for i in range(n_cls)]
        except Exception:
            cls_names = list(reverse_mapping.values())[:len(probabilities)]

        fig_p = go.Figure(go.Bar(
            x=[p * 100 for p in probabilities], y=cls_names, orientation="h",
            marker_color=["#1565C0" if cls_names[i] == disease else "#BBDEFB"
                          for i in range(len(probabilities))],
            text=[f"{p * 100:.1f}%" for p in probabilities], textposition="outside",
        ))
        fig_p.update_layout(
            title="Probability Distribution Across All Classes",
            xaxis_title="Probability (%)", xaxis=dict(range=[0, 115]),
            height=220, margin=dict(l=10, r=60, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="#F7F9FC",
            font={"family": "DM Sans", "color": "#1B2A4A"},
        )
        st.plotly_chart(fig_p, use_container_width=True)

        ALERTS = {
            "Heart Attack": "EMERGENCY: Seek immediate medical attention. Call ambulance now.",
            "Angina":       "Please avoid heavy exercise and consult a cardiologist soon.",
            "Arrhythmia":   "Monitor your heart rhythm regularly. Follow up with a specialist.",
            "No Disease":   "No significant cardiac condition detected. Maintain a healthy lifestyle.",
        }
        alert_box(ALERTS.get(disease, "Please consult a doctor."), info["alert_type"])

        # ── Disease Info ──────────────────────────────────────────────────── #
        st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
        section_header(f"{info['emoji']} About {disease}")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700;font-size:15px;color:#0D47A1;margin-bottom:8px">📖 What is it?</div>
                <div style="font-size:14px;color:#455A64;line-height:1.7">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            li = "".join(f"<li style='margin:5px 0'>{c}</li>" for c in info["causes"])
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700;font-size:15px;color:#0D47A1;margin-bottom:8px">⚡ Key Causes</div>
                <ul style="font-size:14px;color:#455A64;padding-left:18px;line-height:1.7">{li}</ul>
            </div>""", unsafe_allow_html=True)
        with d3:
            pl = "".join(f"<li style='margin:4px 0'>{p}</li>" for p in info["precautions"])
            dl = "".join(f"<li style='margin:4px 0'>{d}</li>" for d in info["diet"])
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700;font-size:15px;color:#0D47A1;margin-bottom:4px">🛡️ Precautions</div>
                <ul style="font-size:13px;color:#455A64;padding-left:18px">{pl}</ul>
                <div style="font-weight:700;font-size:15px;color:#0D47A1;margin:8px 0 4px">🥗 Diet Tips</div>
                <ul style="font-size:13px;color:#455A64;padding-left:18px">{dl}</ul>
            </div>""", unsafe_allow_html=True)

        # ── LIME Explanation ──────────────────────────────────────────────── #
        st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
        section_header("🔬 LIME Explanation", "LIME shows which symptoms most influenced this prediction")

        if not st.session_state.lime_done:
            with st.spinner("Generating LIME explanation..."):
                try:
                    rng = np.random.RandomState(42)
                    n   = 30
                    bg  = np.hstack([
                        rng.randint(20, 90, n).reshape(-1, 1).astype(float),
                        rng.randint(0, 2, n).reshape(-1, 1).astype(float),
                        rng.choice([0, 0.5, 1], size=(n, 11)).astype(float),
                    ])

                    exp_cls_names = [reverse_mapping[i] for i in range(len(probabilities))]

                    lime_exp = LimeTabularExplainer(
                        training_data=bg,
                        feature_names=FEATURE_NAMES,
                        class_names=exp_cls_names,
                        mode="classification",
                        random_state=42,
                    )

                    pred_cls = int(prediction)
                    exp = lime_exp.explain_instance(
                        input_data[0],
                        model.predict_proba,
                        num_features=13,
                        labels=[pred_cls],
                    )

                    lime_df = pd.DataFrame(
                        exp.as_list(label=pred_cls),
                        columns=["Feature", "Weight"],
                    )
                    lime_df = lime_df.sort_values("Weight", key=abs, ascending=True)

                    fig_l = go.Figure(go.Bar(
                        x=lime_df["Weight"],
                        y=lime_df["Feature"],
                        orientation="h",
                        marker_color=["#C62828" if w > 0 else "#1565C0" for w in lime_df["Weight"]],
                        text=[f"{w:+.4f}" for w in lime_df["Weight"]],
                        textposition="outside",
                    ))
                    fig_l.update_layout(
                        title=f"Feature Contributions for: {disease}",
                        xaxis_title="LIME Weight  (+ increases risk  |  - decreases risk)",
                        height=430,
                        margin=dict(l=10, r=70, t=50, b=10),
                        paper_bgcolor="white",
                        plot_bgcolor="#F7F9FC",
                        font={"family": "DM Sans", "color": "#1B2A4A"},
                    )

                    st.session_state.lime_fig  = fig_l
                    st.session_state.lime_done = True

                except Exception as e:
                    st.error(f"LIME explanation error: {e}")

        if st.session_state.lime_fig is not None:
            st.plotly_chart(st.session_state.lime_fig, use_container_width=True)
            st.caption("🔴 Red = increases predicted probability   |   🔵 Blue = decreases predicted probability")

        # ── Doctor Booking ────────────────────────────────────────────────── #
        st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
        section_header("👨‍⚕️ Recommended Cardiologists", f"Doctors available in {city}")

        if not city_doctors:
            st.info("No doctors listed for this city. Please consult a local cardiologist.")
        else:
            for idx, doc in enumerate(city_doctors):
                dk = f"{idx}_{doc['name'].replace(' ', '_')}"

                st.markdown(f"""
                <div class="doc-card">
                    <div style="font-weight:700;font-size:17px;color:#0D47A1">👨‍⚕️ {doc['name']}</div>
                    <div style="font-size:13px;color:#607D8B;margin-top:2px">
                        {doc['specialist']} · {doc['hospital']}
                    </div>
                    <div style="font-size:13px;color:#455A64;margin-top:6px">
                        Experience: <b>{doc['exp']}</b> | Consultation Fee: <b>{doc['fees']}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

                if st.button(f"📅 Book Appointment with {doc['name']}", key=f"book_{dk}"):
                    st.session_state.booking_open[dk] = not st.session_state.booking_open.get(dk, False)

                if st.session_state.booking_open.get(dk, False):
                    st.markdown('<div class="booking-form">', unsafe_allow_html=True)
                    st.markdown(f"**📋 Book with {doc['name']}**")
                    bf1, bf2, bf3 = st.columns(3)
                    with bf1:
                        bk_name = st.text_input(
                            "Patient Name",
                            value=patient_name or "",
                            key=f"bname_{dk}",
                        )
                    with bf2:
                        bk_date = st.date_input(
                            "Appointment Date",
                            min_value=date.today(),
                            key=f"bdate_{dk}",
                        )
                    with bf3:
                        bk_time = st.selectbox(
                            "Time Slot",
                            ["09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM",
                             "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM"],
                            key=f"btime_{dk}",
                        )

                    if st.button("✅ Confirm Appointment", key=f"conf_{dk}"):
                        st.session_state.booking_confirmed[dk] = {
                            "name": bk_name.strip() or "Patient",
                            "doc":  doc["name"],
                            "date": bk_date.strftime("%d %B %Y"),
                            "time": bk_time,
                        }
                        st.session_state.booking_open[dk] = False

                    st.markdown('</div>', unsafe_allow_html=True)

                conf = st.session_state.booking_confirmed.get(dk)
                if conf:
                    st.success(
                        f"🎉 Appointment confirmed! "
                        f"Patient: {conf['name']} | "
                        f"Doctor: {conf['doc']} | "
                        f"Date: {conf['date']} | "
                        f"Time: {conf['time']}"
                    )

                st.markdown(
                    "<hr style='margin:10px 0;border-color:#E8EEF7'>",
                    unsafe_allow_html=True,
                )

        # ── PDF Report ────────────────────────────────────────────────────── #
        st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
        section_header("📄 Download Health Report")

        try:
            from fpdf import FPDF

            class PDF(FPDF):
                def header(self):
                    self.set_fill_color(21, 101, 192)
                    self.rect(0, 0, 210, 32, "F")
                    self.set_font("Helvetica", "B", 17)
                    self.set_text_color(255, 255, 255)
                    self.set_y(8)
                    self.cell(0, 10, "CardioAI - Heart Disease Detection Report", ln=True, align="C")
                    self.set_font("Helvetica", "", 10)
                    self.cell(0, 8, "AI-Powered Cardiac Health Assessment", ln=True, align="C")
                    self.set_text_color(0, 0, 0)
                    self.ln(8)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(128, 128, 128)
                    self.cell(
                        0, 10,
                        "Generated by CardioAI | For informational purposes only. "
                        "Not a substitute for professional medical advice.",
                        align="C",
                    )

            def pdf_section(pdf, title):
                pdf.set_fill_color(239, 244, 255)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(21, 101, 192)
                pdf.cell(0, 9, safe_str(title), ln=True, fill=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(1)

            def pdf_row(pdf, label, value):
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(65, 7, safe_str(label) + ":", ln=False)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, safe_str(value), ln=True)

            def pdf_bullet(pdf, text):
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(8, 7, "*", ln=False)
                pdf.cell(0, 7, safe_str(text), ln=True)

            def generate_pdf():
                pdf = PDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=18)

                # Timestamp
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(128, 128, 128)
                pdf.cell(
                    0, 6,
                    f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
                    ln=True,
                )
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)

                # Patient Information
                pdf_section(pdf, "  Patient Information")
                pdf_row(pdf, "Name",   patient_name or "N/A")
                pdf_row(pdf, "Age",    f"{age} years")
                pdf_row(pdf, "Gender", sex_label)
                pdf_row(pdf, "City",   city)
                pdf.ln(3)

                # Prediction Result
                pdf_section(pdf, "  Prediction Result")
                pdf_row(pdf, "Detected Condition", disease)
                pdf_row(pdf, "AI Confidence",      f"{confidence:.2f}%")
                pdf.ln(3)

                # Symptom Inputs
                pdf_section(pdf, "  Symptom Inputs")
                SYM_LABELS = [
                    "Chest Pain", "Shortness of Breath", "Chest Tightness",
                    "Palpitations", "Irregular Heartbeat", "Dizziness",
                    "Fatigue", "Leg Swelling", "High BP",
                    "High Cholesterol", "Exercise Pain",
                ]
                for sl, sk in zip(SYM_LABELS, SKEYS):
                    pdf_row(pdf, sl, raw_vals[sk])
                pdf.ln(3)

                # Precautions
                pdf_section(pdf, "  Precautions")
                for p in info["precautions"]:
                    pdf_bullet(pdf, p)
                pdf.ln(3)

                # Diet Recommendations
                pdf_section(pdf, "  Diet Recommendations")
                for d in info["diet"]:
                    pdf_bullet(pdf, d)
                pdf.ln(3)

                # Recommended Doctor
                if city_doctors:
                    pdf_section(pdf, "  Recommended Doctor")
                    doc = city_doctors[0]
                    pdf_row(pdf, "Name",      doc["name"])
                    pdf_row(pdf, "Specialist", doc["specialist"])
                    pdf_row(pdf, "Hospital",   doc["hospital"])
                    pdf_row(pdf, "Fees",       doc["fees"])
                    pdf.ln(3)

                # Disclaimer
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(128, 128, 128)
                pdf.multi_cell(
                    0, 5,
                    "DISCLAIMER: This report is generated by an AI system for informational "
                    "purposes only. It does not constitute medical advice. Please consult a "
                    "qualified cardiologist for proper diagnosis and treatment.",
                )

                return bytes(pdf.output())

            pdf_bytes = generate_pdf()
            fname = f"CardioAI_Report_{(patient_name or 'Patient').replace(' ', '_')}.pdf"

            st.download_button(
                label="📥  Download PDF Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
            )
            st.caption(
                "Includes: patient info, prediction, symptoms, precautions, "
                "diet tips, and doctor recommendation."
            )

        except ImportError:
            st.warning("Install fpdf2 to enable PDF reports:  pip install fpdf2")
        except Exception as e:
            st.error(f"PDF generation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    if not MODEL_LOADED:
        st.error("Model files not found.")
        st.stop()

    section_header("📊 Model Insights", "Real metrics from your trained model evaluation")

    # ── Confusion Matrix ───────────────────────────────────────────────────── #
    st.markdown("### 📉 Confusion Matrix")
    st.caption("Actual model evaluation on test set — 1600 samples")

    fig_cm = px.imshow(
        REAL_CM,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=CM_LABELS, y=CM_LABELS,
        color_continuous_scale=[[0, "#EFF4FF"], [0.4, "#64B5F6"], [1, "#0D47A1"]],
        text_auto=True, zmin=0, zmax=int(REAL_CM.max()),
        title="Confusion Matrix — Test Set (1,600 samples)",
    )
    fig_cm.update_traces(textfont=dict(size=15, color="white"))
    fig_cm.update_layout(
        height=430, font={"family": "DM Sans", "size": 13},
        paper_bgcolor="white",
        title_font=dict(size=16, color="#0D47A1"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class summary boxes
    cm1, cm2, cm3, cm4 = st.columns(4)
    for col, cls, color in zip(
        [cm1, cm2, cm3, cm4], CM_LABELS,
        ["#4CAF50", "#F44336", "#2196F3", "#FF9800"],
    ):
        i   = CM_LABELS.index(cls)
        tp  = REAL_CM[i, i]
        sup = REAL_CM[i].sum()
        with col:
            st.markdown(f"""
            <div class="metric-box" style="border-left:4px solid {color}">
                <div style="font-size:11px;color:#607D8B;text-transform:uppercase">{cls}</div>
                <div class="val" style="font-size:22px">{tp}/{sup}</div>
                <div class="lbl">Correct / Total</div>
            </div>""", unsafe_allow_html=True)

    # ── Classification Report ─────────────────────────────────────────────── #
    st.markdown("### 📋 Classification Report")
    st.caption("Exact output from your model evaluation")

    rows = []
    for cls in ["No Disease", "Heart Attack", "Arrhythmia", "Angina",
                "accuracy", "macro avg", "weighted avg"]:
        r = REAL_REPORT[cls]
        rows.append({
            "Class":     cls,
            "Precision": f"{r['precision']:.2f}" if r["precision"] is not None else "",
            "Recall":    f"{r['recall']:.2f}"    if r["recall"]    is not None else "",
            "F1-Score":  f"{r['f1']:.2f}",
            "Support":   int(r["support"]),
        })
    rdf = pd.DataFrame(rows).set_index("Class")

    def color_cell(val):
        try:
            v = float(val)
            if v >= 0.98:
                return "background-color:#C8E6C9;color:#1B5E20;font-weight:700"
            if v >= 0.95:
                return "background-color:#DCEDC8;color:#33691E;font-weight:600"
            if v >= 0.90:
                return "background-color:#FFF9C4;color:#F57F17"
            return "background-color:#FFCCBC;color:#BF360C"
        except Exception:
            return ""

    st.dataframe(
        rdf.style.applymap(color_cell, subset=["Precision", "Recall", "F1-Score"]),
        use_container_width=True, height=280,
    )

    # Overall metrics
    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    am1, am2, am3 = st.columns(3)
    for col, (ico, val, lbl) in zip([am1, am2, am3], [
        ("🎯", "97%", "Overall Accuracy"),
        ("📊", "0.97", "Macro F1-Score"),
        ("🧪", "1,600", "Test Samples"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:28px">{ico}</div>
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Feature Importance ─────────────────────────────────────────────────── #
    st.markdown("### 🌟 Feature Importance")
    try:
        rf = None
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    rf = est
                    break
        elif hasattr(model, "feature_importances_"):
            rf = model

        if rf is not None:
            fi_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": rf.feature_importances_})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#EFF4FF"], [0.5, "#64B5F6"], [1, "#0D47A1"]],
                    showscale=True, colorbar=dict(title="Importance"),
                ),
                text=[f"{v:.4f}" for v in fi_df["Importance"]], textposition="outside",
            ))
            fig_fi.update_layout(
                title="Feature Importances (Random Forest Component)",
                xaxis_title="Importance Score",
                height=460, margin=dict(l=10, r=80, t=50, b=10),
                paper_bgcolor="white", plot_bgcolor="#F7F9FC",
                font={"family": "DM Sans", "color": "#1B2A4A"},
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances not available (no RF component with feature_importances_ found).")
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")

    # ── Model Architecture Cards ──────────────────────────────────────────── #
    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    section_header("🤖 Model Architecture")
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (ico, title, desc) in zip([mc1, mc2, mc3, mc4], [
        ("🌲", "Random Forest",    "Tuned with GridSearchCV\nn_estimators, max_depth"),
        ("🧮", "SVM",              "RBF kernel\nC and gamma optimized"),
        ("🗳️", "Voting Classifier", "Soft voting ensemble\nRF + SVM combined"),
        ("🔬", "LIME",             "Local Interpretable\nModel-agnostic Explanations"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-box" style="text-align:left;padding:18px">
                <div style="font-size:28px">{ico}</div>
                <div style="font-weight:700;font-size:14px;color:#0D47A1;margin-top:6px">{title}</div>
                <div style="font-size:12px;color:#607D8B;margin-top:4px;white-space:pre-line">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    section_header("ℹ️ About CardioAI")

    st.markdown("""
    <div class="card-blue">
        <div style="font-family:'DM Serif Display',serif;font-size:28px;margin-bottom:8px">
            2nd Year B.Tech Project 2025-26
        </div>
        <div style="opacity:0.85;font-size:15px;line-height:1.8">
            CardioAI is an AI-powered heart disease detection system built as a 2nd Year Project.
            It uses a trained Voting Classifier (Random Forest + SVM) ensemble to predict cardiac
            conditions from 13 clinical symptom features, with LIME-based explainability and
            integrated doctor recommendation and appointment booking.
        </div>
    </div>""", unsafe_allow_html=True)

    section_header("👥 Team")
    team = [
        ("Atul Krishna",      "Data Engineer and Team Lead",            "📊", "Dataset Preprocessing, Feature Selection, Team Management"),
        ("Raza Ansari",       "Frontend and UI Lead",                   "🎨", "Streamlit UI, CSS styling, dashboard design"),
        ("Pulkit Goyal",      "Machine Learning and Model Integration", "🧠", "Model training, LIME integration, pipeline"),
        ("Saurabh Kumar Jha", "Junior Frontend Developer, Research",    "📝", "Light Frontend work, Literature survey, report, presentation"),
    ]
    tc1, tc2 = st.columns(2)
    for i, (name, role, ico, detail) in enumerate(team):
        col = tc1 if i % 2 == 0 else tc2
        with col:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;align-items:center;gap:14px">
                    <div style="width:52px;height:52px;border-radius:50%;flex-shrink:0;
                                background:linear-gradient(135deg,#1565C0,#1976D2);
                                display:flex;align-items:center;justify-content:center;
                                font-size:24px">{ico}</div>
                    <div>
                        <div style="font-weight:700;font-size:16px;color:#0D47A1">{name}</div>
                        <div style="font-size:13px;color:#607D8B;font-weight:500">{role}</div>
                        <div style="font-size:12px;color:#78909C;margin-top:2px">{detail}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    section_header("🛠️ Technology Stack")
    techs = [
        ("Python 3.10+",   "Core language"),         ("Streamlit",  "Web dashboard"),
        ("scikit-learn",   "ML models and metrics"),  ("LIME",       "Model explainability"),
        ("Plotly",         "Interactive charts"),     ("fpdf2",      "PDF generation"),
        ("pandas / numpy", "Data processing"),        ("joblib",     "Model serialization"),
    ]
    for row in [techs[i:i+4] for i in range(0, len(techs), 4)]:
        cols = st.columns(4)
        for col, (name, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-weight:700;font-size:14px;color:#0D47A1">{name}</div>
                    <div class="lbl">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#90A4AE;font-size:13px;padding:20px 0">
        CardioAI | 2nd Year Project 2025-26<br>
        This tool is for educational and informational purposes only.
        Not a substitute for professional medical advice.
    </div>""", unsafe_allow_html=True)
