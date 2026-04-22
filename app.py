import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import json
import io
import time
from pathlib import Path
from datetime import datetime
import warnings

import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AquaShield India Premium",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "test_history.json"

# =========================================================
# INDIA STATES + MAHARASHTRA DISTRICTS
# =========================================================
INDIA_STATES_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

MAHARASHTRA_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana",
    "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna",
    "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad",
    "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha",
    "Washim", "Yavatmal"
]

# =========================================================
# TEXTS
# =========================================================
TEXTS = {
    "English": {
        "title": "💧 AquaShield India Premium",
        "subtitle": "Water Disease Prediction, Safety Analysis & Health Advisory Dashboard",
        "tagline": "Clear visuals, smart alerts, India-ready geography, teacher-friendly premium design",
        "dashboard": "🏠 Dashboard",
        "test": "🧪 Water Quality Test",
        "batch": "📂 Batch Analysis",
        "analytics": "📈 Live Analytics",
        "history": "🕒 Test History",
        "bot": "🤖 Awareness Bot",
        "about": "ℹ️ About System",
        "deploy": "🚀 Deploy Guide",
        "navigation": "Navigation",
        "stats": "System Stats",
        "total_tests": "Total Tests",
        "alerts": "Active Alerts",
        "accuracy": "Model Accuracy",
        "features": "Features",
        "fill": "Fill the details below and click Predict & Analyze.",
        "location": "📍 Location & Source Details",
        "params": "💧 Water Quality Parameters",
        "predict": "🚀 Predict & Analyze",
        "results": "Detailed Prediction Results",
        "recommend": "🧠 Smart Recommendations",
        "safety": "⚠️ Water Safety Analysis",
        "history_title": "Recent Test Activity",
        "prob_chart": "Disease Probability Chart",
        "risk_chart": "Overall Risk Gauge",
        "param_chart": "Parameter Profile",
        "trend_chart": "Risk Trend Over Time",
        "health_advisory": "🩺 Health Advisory Cards",
        "export_pdf": "⬇️ Export PDF Report",
        "download_csv": "⬇️ Download CSV",
        "upload_csv": "Upload CSV",
        "run_batch": "Run Batch Prediction",
        "required_cols": "Required CSV Columns",
        "single_dashboard": "🎯 Prediction Dashboard",
        "clear_history": "🗑️ Clear History",
        "models_missing": "Required model files were not found in the models folder.",
        "prediction_failed": "Prediction failed",
        "batch_failed": "Batch analysis failed",
        "safe_msg": "All major parameters appear within comparatively safer ranges.",
        "unsafe_msg": "Unsafe parameters detected",
        "no_history": "No history available yet.",
        "state_ui": "Selected State",
        "district_ui": "District",
        "model_state": "Model State",
        "mh_note": "Maharashtra selected — district dropdown enabled.",
        "unsupported_state_note": "This state is outside the trained model geography. Prediction may not be available unless you select a supported model state.",
        "deploy_text": "Push your project to GitHub, then connect the repository on Streamlit Community Cloud and select app.py.",
        "geo_card_title": "🗺️ Geography Intelligence",
        "geo_card_text": "All India states are available in UI. Maharashtra includes district-level selection for richer demo presentation.",
        "supported_state_info": "Supported model states",
        "batch_summary": "Batch Risk Distribution",
        "loader_1": "Loading dashboard engine...",
        "loader_2": "Loading ML prediction modules...",
        "loader_3": "Preparing charts, health cards and reports...",
        "loader_4": "System ready.",
        "bot_title": "🤖 Water Safety Awareness Bot",
        "bot_desc": "This chatbot helps users learn about safe drinking water, contamination prevention, hygiene, and emergency precautions.",
        "bot_tip_1": "Ask: Is my water safe to drink?",
        "bot_tip_2": "Ask: How to prevent cholera and typhoid?",
        "bot_tip_3": "Ask: What should I do if water is contaminated?",
        "bot_open": "🌐 Open Bot in New Tab",
        "bot_embed_note": "If the embedded bot does not load, use the button below to open it in a new tab.",
        "water_score": "Water Safety Score",
        "traffic_title": "🚦 Traffic Light Risk Widget",
    },
    "Marathi": {
        "title": "💧 AquaShield India Premium",
        "subtitle": "पाणीजन्य रोग भाकीत, सुरक्षा विश्लेषण आणि आरोग्य सूचना डॅशबोर्ड",
        "tagline": "स्पष्ट visuals, स्मार्ट alerts, India-ready geography आणि premium design",
        "dashboard": "🏠 डॅशबोर्ड",
        "test": "🧪 पाणी गुणवत्ता चाचणी",
        "batch": "📂 बॅच विश्लेषण",
        "analytics": "📈 लाईव्ह अॅनालिटिक्स",
        "history": "🕒 चाचणी इतिहास",
        "bot": "🤖 Awareness Bot",
        "about": "ℹ️ सिस्टम माहिती",
        "deploy": "🚀 डिप्लॉय मार्गदर्शक",
        "navigation": "नेव्हिगेशन",
        "stats": "सिस्टम आकडे",
        "total_tests": "एकूण चाचण्या",
        "alerts": "सक्रिय अलर्ट",
        "accuracy": "मॉडेल अचूकता",
        "features": "फीचर्स",
        "fill": "खालील माहिती भरा आणि Predict & Analyze वर क्लिक करा.",
        "location": "📍 ठिकाण आणि स्रोत माहिती",
        "params": "💧 पाणी गुणवत्ता मापदंड",
        "predict": "🚀 Predict & Analyze",
        "results": "सविस्तर निकाल",
        "recommend": "🧠 स्मार्ट शिफारसी",
        "safety": "⚠️ पाणी सुरक्षा विश्लेषण",
        "history_title": "अलीकडील चाचणी नोंदी",
        "prob_chart": "रोग संभाव्यता चार्ट",
        "risk_chart": "एकूण जोखीम गेज",
        "param_chart": "मापदंड प्रोफाइल",
        "trend_chart": "वेळेनुसार जोखीम ट्रेंड",
        "health_advisory": "🩺 आरोग्य सूचना कार्ड्स",
        "export_pdf": "⬇️ PDF रिपोर्ट डाउनलोड",
        "download_csv": "⬇️ CSV डाउनलोड",
        "upload_csv": "CSV अपलोड करा",
        "run_batch": "बॅच प्रेडिक्शन चालवा",
        "required_cols": "आवश्यक CSV कॉलम्स",
        "single_dashboard": "🎯 प्रेडिक्शन डॅशबोर्ड",
        "clear_history": "🗑️ इतिहास साफ करा",
        "models_missing": "models folder मध्ये आवश्यक model files सापडल्या नाहीत.",
        "prediction_failed": "प्रेडिक्शन फेल झाले",
        "batch_failed": "बॅच विश्लेषण फेल झाले",
        "safe_msg": "मुख्य मापदंड तुलनेने सुरक्षित मर्यादेत आहेत.",
        "unsafe_msg": "असुरक्षित मापदंड आढळले",
        "no_history": "अजून history उपलब्ध नाही.",
        "state_ui": "निवडलेले राज्य",
        "district_ui": "जिल्हा",
        "model_state": "मॉडेल राज्य",
        "mh_note": "Maharashtra निवडले आहे — district dropdown enabled.",
        "unsupported_state_note": "हे राज्य trained model geography मध्ये नाही. Supported model state निवडल्याशिवाय prediction reliable नसेल.",
        "deploy_text": "Project GitHub वर push करा, मग Streamlit Community Cloud वर repo connect करून app.py निवडा.",
        "geo_card_title": "🗺️ Geography Intelligence",
        "geo_card_text": "UI मध्ये सर्व India states उपलब्ध आहेत. Maharashtra साठी district-level selection दिले आहे.",
        "supported_state_info": "मॉडेलला support असलेली राज्ये",
        "batch_summary": "बॅच जोखीम वितरण",
        "loader_1": "Dashboard engine load होत आहे...",
        "loader_2": "ML prediction modules load होत आहेत...",
        "loader_3": "Charts, health cards आणि reports तयार होत आहेत...",
        "loader_4": "System तयार आहे.",
        "bot_title": "🤖 Water Safety Awareness Bot",
        "bot_desc": "हा chatbot सुरक्षित पिण्याचे पाणी, contamination prevention, hygiene आणि emergency precautions बद्दल माहिती देतो.",
        "bot_tip_1": "विचार: हे पाणी पिण्यास सुरक्षित आहे का?",
        "bot_tip_2": "विचार: cholera आणि typhoid कसे टाळायचे?",
        "bot_tip_3": "विचार: पाणी contaminated असेल तर काय करायचे?",
        "bot_open": "🌐 Bot नवीन tab मध्ये उघडा",
        "bot_embed_note": "जर embedded bot load झाला नाही तर खालील button वापरून new tab मध्ये उघडा.",
        "water_score": "Water Safety Score",
        "traffic_title": "🚦 Traffic Light Risk Widget",
    },
    "Hindi": {
        "title": "💧 AquaShield India Premium",
        "subtitle": "जल-जनित रोग भविष्यवाणी, सुरक्षा विश्लेषण और स्वास्थ्य सलाह डैशबोर्ड",
        "tagline": "स्पष्ट visuals, smart alerts, India-ready geography और premium design",
        "dashboard": "🏠 Dashboard",
        "test": "🧪 Water Quality Test",
        "batch": "📂 Batch Analysis",
        "analytics": "📈 Live Analytics",
        "history": "🕒 Test History",
        "bot": "🤖 Awareness Bot",
        "about": "ℹ️ About System",
        "deploy": "🚀 Deploy Guide",
        "navigation": "Navigation",
        "stats": "System Stats",
        "total_tests": "Total Tests",
        "alerts": "Active Alerts",
        "accuracy": "Model Accuracy",
        "features": "Features",
        "fill": "नीचे सभी विवरण भरें और Predict & Analyze पर क्लिक करें.",
        "location": "📍 Location & Source Details",
        "params": "💧 Water Quality Parameters",
        "predict": "🚀 Predict & Analyze",
        "results": "Detailed Results",
        "recommend": "🧠 Smart Recommendations",
        "safety": "⚠️ Water Safety Analysis",
        "history_title": "Recent Test Activity",
        "prob_chart": "Disease Probability Chart",
        "risk_chart": "Overall Risk Gauge",
        "param_chart": "Parameter Profile",
        "trend_chart": "Risk Trend Over Time",
        "health_advisory": "🩺 Health Advisory Cards",
        "export_pdf": "⬇️ Export PDF Report",
        "download_csv": "⬇️ Download CSV",
        "upload_csv": "Upload CSV",
        "run_batch": "Run Batch Prediction",
        "required_cols": "Required CSV Columns",
        "single_dashboard": "🎯 Prediction Dashboard",
        "clear_history": "🗑️ Clear History",
        "models_missing": "models folder में required model files नहीं मिलीं.",
        "prediction_failed": "Prediction failed",
        "batch_failed": "Batch analysis failed",
        "safe_msg": "मुख्य पैरामीटर अपेक्षाकृत सुरक्षित सीमा में हैं.",
        "unsafe_msg": "Unsafe parameters detected",
        "no_history": "अभी history उपलब्ध नहीं है.",
        "state_ui": "Selected State",
        "district_ui": "District",
        "model_state": "Model State",
        "mh_note": "Maharashtra selected — district dropdown enabled.",
        "unsupported_state_note": "यह state trained model geography के बाहर है. Reliable prediction के लिए supported model state चुनें.",
        "deploy_text": "Project GitHub पर push करें, फिर Streamlit Community Cloud पर repo connect करके app.py चुनें.",
        "geo_card_title": "🗺️ Geography Intelligence",
        "geo_card_text": "UI में सभी India states available हैं. Maharashtra के लिए district-level selection दिया गया है.",
        "supported_state_info": "Supported model states",
        "batch_summary": "Batch Risk Distribution",
        "loader_1": "Dashboard engine load हो रहा है...",
        "loader_2": "ML prediction modules load हो रहे हैं...",
        "loader_3": "Charts, health cards और reports तैयार हो रहे हैं...",
        "loader_4": "System ready.",
        "bot_title": "🤖 Water Safety Awareness Bot",
        "bot_desc": "यह chatbot safe drinking water, contamination prevention, hygiene और emergency precautions के बारे में जानकारी देता है.",
        "bot_tip_1": "पूछें: क्या यह पानी पीने के लिए safe है?",
        "bot_tip_2": "पूछें: cholera और typhoid को कैसे रोका जाए?",
        "bot_tip_3": "पूछें: अगर पानी contaminated हो तो क्या करें?",
        "bot_open": "🌐 Bot को new tab में खोलें",
        "bot_embed_note": "अगर embedded bot load नहीं होता है, तो नीचे वाला button use करें.",
        "water_score": "Water Safety Score",
        "traffic_title": "🚦 Traffic Light Risk Widget",
    },
}

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 20%),
            radial-gradient(circle at top right, rgba(167,139,250,0.16), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #eef4ff 45%, #f8fbff 100%);
        color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 45%, #7c3aed 100%);
        border-radius: 26px;
        padding: 1.7rem;
        color: white;
        box-shadow: 0 18px 40px rgba(37, 99, 235, 0.20);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.97;
    }
    .metric-tile {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(37,99,235,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 12px 26px rgba(15,23,42,0.06);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #1d4ed8;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .result-low {
        background: linear-gradient(90deg, #10b981, #14b8a6);
        color: white;
        padding: 1rem;
        border-radius: 18px;
        font-weight: 800;
    }
    .result-medium {
        background: linear-gradient(90deg, #f59e0b, #f97316);
        color: white;
        padding: 1rem;
        border-radius: 18px;
        font-weight: 800;
    }
    .result-high {
        background: linear-gradient(90deg, #ef4444, #ec4899);
        color: white;
        padding: 1rem;
        border-radius: 18px;
        font-weight: 800;
    }
    .advisory-card {
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 12px 24px rgba(15,23,42,0.10);
    }
    .advisory-red { background: linear-gradient(135deg, #ef4444, #f97316); }
    .advisory-yellow { background: linear-gradient(135deg, #facc15, #f59e0b); color: #111827; }
    .advisory-green { background: linear-gradient(135deg, #10b981, #06b6d4); }
    .deploy-box {
        background: linear-gradient(135deg, #eff6ff, #f5f3ff);
        border: 1px solid rgba(99,102,241,0.16);
        border-radius: 18px;
        padding: 1rem;
        color: #0f172a;
        box-shadow: 0 12px 24px rgba(15,23,42,0.06);
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
    }
    .splash-wrap {
        background: linear-gradient(135deg, #e0f2fe 0%, #eef2ff 55%, #f5f3ff 100%);
        border: 1px solid rgba(99,102,241,0.14);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 18px 36px rgba(15,23,42,0.08);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .splash-title {
        font-size: 2rem;
        font-weight: 900;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .splash-sub {
        font-size: 1rem;
        color: #334155;
    }
    .traffic-wrap {
        display:flex;
        gap:14px;
        align-items:center;
        margin: 0.5rem 0 1rem 0;
    }
    .traffic-light {
        width:26px;
        height:26px;
        border-radius:50%;
        border:2px solid rgba(15,23,42,0.18);
        box-shadow: 0 4px 10px rgba(15,23,42,0.12);
    }
    .traffic-on-green { background:#10b981; }
    .traffic-on-yellow { background:#f59e0b; }
    .traffic-on-red { background:#ef4444; }
    .traffic-off { background:#e2e8f0; }
    .score-box {
        background: linear-gradient(135deg, #ffffff, #eff6ff);
        border: 1px solid rgba(37,99,235,0.12);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 22px rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 10px 22px rgba(15,23,42,0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 900;
    }
    .stDataFrame, .stTable {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def safe_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_metadata():
    for path in [MODELS_DIR / "metadata.json", MODELS_DIR / "model_metadata.json"]:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None

@st.cache_resource
def load_assets():
    metadata = load_metadata()
    if metadata is None:
        return None

    enc_path = MODELS_DIR / "label_encoders.pkl"
    feat_path = MODELS_DIR / "feature_names.pkl"

    if not enc_path.exists() or not feat_path.exists():
        return None

    encoders = safe_load_pickle(enc_path)
    features = safe_load_pickle(feat_path)

    diseases = ["cholera", "typhoid", "dysentery", "hepatitis_a", "overall"]
    models = {}
    scalers = {}

    for d in diseases:
        mp = MODELS_DIR / f"{d}_model.pkl"
        sp = MODELS_DIR / f"{d}_scaler.pkl"
        if mp.exists():
            models[d] = safe_load_pickle(mp)
        if sp.exists():
            scalers[d] = safe_load_pickle(sp)

    if not models or not scalers:
        return None

    return {
        "metadata": metadata,
        "encoders": encoders,
        "features": features,
        "models": models,
        "scalers": scalers,
    }

def initialize_history():
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def load_history():
    initialize_history()
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:500]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_classes(encoders, key):
    return list(encoders[key].classes_) if key in encoders else []

def safe_transform(encoders, key, value):
    classes = list(encoders[key].classes_)
    if value not in classes:
        raise ValueError(f"Unknown value '{value}' in {key}")
    return int(encoders[key].transform([value])[0])

def prepare_input_dataframe(input_dict, features):
    row = {}
    for feat in features:
        row[feat] = input_dict.get(feat, 0)
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0)

def predict_all(input_df, models, scalers):
    results = {}
    for disease, model in models.items():
        scaler = scalers[disease]
        scaled = scaler.transform(input_df)
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1]) if hasattr(model, "predict_proba") else float(pred)
        results[disease] = {"prediction": pred, "probability": prob}
    return results

def get_risk(prob):
    if prob < 0.35:
        return "Low Risk", "low"
    if prob < 0.70:
        return "Moderate Risk", "medium"
    return "High Risk", "high"

def risk_box(label, level):
    cls = "result-low" if level == "low" else "result-medium" if level == "medium" else "result-high"
    return f'<div class="{cls}">🚨 Overall Outbreak Alert: {label}</div>'

def parameter_warnings(values):
    warnings_list = []
    if values["ph"] < 6.5 or values["ph"] > 8.5:
        warnings_list.append("pH is outside safer range")
    if values["turbidity_ntu"] > 5:
        warnings_list.append("Turbidity is high")
    if values["bod_mg_l"] > 3:
        warnings_list.append("BOD is high")
    if values["fecal_coliform_mpn"] > 10:
        warnings_list.append("Fecal coliform is dangerously high")
    if values["total_coliform_mpn"] > 50:
        warnings_list.append("Total coliform is high")
    if values["arsenic_ug_l"] > 10:
        warnings_list.append("Arsenic exceeds safe limit")
    if values["iron_mg_l"] > 0.3:
        warnings_list.append("Iron exceeds recommended limit")
    if values["nitrate_mg_l"] > 45:
        warnings_list.append("Nitrate is too high")
    if values["fluoride_mg_l"] > 1.5:
        warnings_list.append("Fluoride is too high")
    if values["sanitation_access_percent"] < 50:
        warnings_list.append("Sanitation access is low")
    return warnings_list

def generate_recommendations(results, warnings_list):
    recs = []
    overall_prob = results["overall"]["probability"]

    if overall_prob >= 0.70:
        recs.extend([
            "🚨 Immediate public health alert recommended",
            "🔥 Boil water before drinking",
            "🧴 Start chlorination and disinfection",
            "🏥 Increase local health monitoring",
        ])
    elif overall_prob >= 0.35:
        recs.extend([
            "⚠️ Water retesting is advised",
            "💧 Use filtered or boiled water temporarily",
            "🧼 Improve sanitation and hygiene practices",
        ])
    else:
        recs.extend([
            "✅ Continue regular monitoring",
            "💙 Current input looks comparatively safer",
        ])

    if results["cholera"]["probability"] >= 0.60:
        recs.append("🦠 Cholera risk elevated: ensure chlorination and safe storage")
    if results["typhoid"]["probability"] >= 0.60:
        recs.append("🤒 Typhoid risk elevated: isolate unsafe drinking source")
    if results["dysentery"]["probability"] >= 0.60:
        recs.append("🧫 Dysentery risk elevated: improve hygiene and wastewater handling")
    if results["hepatitis_a"]["probability"] >= 0.60:
        recs.append("💉 Hepatitis A risk elevated: strengthen sanitation awareness")
    if warnings_list:
        recs.append("⚠️ Unsafe water parameters detected. Immediate recheck recommended")
    return recs

def calculate_water_safety_score(test_params, overall_prob):
    score = 100

    if test_params["ph"] < 6.5 or test_params["ph"] > 8.5:
        score -= 8
    if test_params["turbidity_ntu"] > 5:
        score -= 10
    if test_params["bod_mg_l"] > 3:
        score -= 8
    if test_params["fecal_coliform_mpn"] > 10:
        score -= 15
    if test_params["total_coliform_mpn"] > 50:
        score -= 10
    if test_params["arsenic_ug_l"] > 10:
        score -= 12
    if test_params["iron_mg_l"] > 0.3:
        score -= 6
    if test_params["nitrate_mg_l"] > 45:
        score -= 8
    if test_params["fluoride_mg_l"] > 1.5:
        score -= 7
    if test_params["sanitation_access_percent"] < 50:
        score -= 8

    score -= int(overall_prob * 30)
    return max(0, min(100, score))

def build_pdf_report(test_inputs, results, warnings_list, recs, safety_score):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, y, "AquaShield India Premium - Water Risk Report")
    y -= 26

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18
    pdf.drawString(40, y, f"Water Safety Score: {safety_score}/100")
    y -= 22

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Input Summary")
    y -= 18

    pdf.setFont("Helvetica", 10)
    for key, value in test_inputs.items():
        line = f"{key}: {value}"
        pdf.drawString(50, y, line[:110])
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 50

    y -= 8
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Prediction Results")
    y -= 18

    pdf.setFont("Helvetica", 10)
    for disease, data in results.items():
        txt = f"{disease}: probability {data['probability']*100:.2f}% | prediction {'Alert' if data['prediction']==1 else 'Normal'}"
        pdf.drawString(50, y, txt[:110])
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 50

    y -= 8
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Safety Warnings")
    y -= 18

    pdf.setFont("Helvetica", 10)
    if warnings_list:
        for w in warnings_list:
            lines = simpleSplit(f"- {w}", "Helvetica", 10, 500)
            for line in lines:
                pdf.drawString(50, y, line)
                y -= 14
                if y < 80:
                    pdf.showPage()
                    y = height - 50
    else:
        pdf.drawString(50, y, "No major warnings.")
        y -= 14

    y -= 8
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Recommendations")
    y -= 18

    pdf.setFont("Helvetica", 10)
    for r in recs:
        lines = simpleSplit(f"- {r}", "Helvetica", 10, 500)
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 14
            if y < 80:
                pdf.showPage()
                y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer

def animated_counter(label, value, suffix=""):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div style="font-size:0.95rem;color:#475569;font-weight:700;">{label}</div>
            <div style="font-size:2rem;color:#0f172a;font-weight:900;margin-top:0.2rem;">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_header(t):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{t["title"]}</div>
            <div class="hero-sub">{t["subtitle"]}</div>
            <div class="hero-sub">{t["tagline"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def advisory_card(title, text, level):
    cls = "advisory-green"
    if level == "red":
        cls = "advisory-red"
    elif level == "yellow":
        cls = "advisory-yellow"

    st.markdown(
        f"""
        <div class="advisory-card {cls}">
            <div style="font-size:1rem;font-weight:800;margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:0.95rem; white-space:pre-line;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def traffic_light_widget(level, title):
    green = "traffic-on-green" if level == "low" else "traffic-off"
    yellow = "traffic-on-yellow" if level == "medium" else "traffic-off"
    red = "traffic-on-red" if level == "high" else "traffic-off"

    st.markdown(
        f"""
        <div class="score-box">
            <div style="font-size:1rem;font-weight:800;color:#1d4ed8;margin-bottom:0.7rem;">{title}</div>
            <div class="traffic-wrap">
                <div class="traffic-light {green}"></div>
                <div class="traffic-light {yellow}"></div>
                <div class="traffic-light {red}"></div>
            </div>
            <div style="font-weight:700;color:#334155;">Current Signal: {level.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# CHARTS
# =========================================================
def build_probability_chart(results, t):
    df = pd.DataFrame({
        "Disease": ["Cholera", "Typhoid", "Dysentery", "Hepatitis A", "Overall"],
        "Probability": [
            results["cholera"]["probability"] * 100,
            results["typhoid"]["probability"] * 100,
            results["dysentery"]["probability"] * 100,
            results["hepatitis_a"]["probability"] * 100,
            results["overall"]["probability"] * 100,
        ]
    })
    fig = px.bar(
        df,
        x="Disease",
        y="Probability",
        color="Disease",
        text="Probability",
        title=t["prob_chart"],
        color_discrete_sequence=["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        font=dict(color="#0f172a"),
        showlegend=False,
        title_font=dict(size=20)
    )
    return fig

def build_gauge_chart(prob, t):
    value = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'color': '#0f172a', 'size': 34}},
        title={'text': t["risk_chart"], 'font': {'color': '#0f172a', 'size': 22}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#334155'},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 35], 'color': "#10b981"},
                {'range': [35, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#ef4444"},
            ],
            'threshold': {
                'line': {'color': "#111827", 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a"),
        margin=dict(l=20, r=20, t=70, b=20)
    )
    return fig

def build_parameter_chart(values, t):
    labels = ["pH", "Turbidity", "BOD", "Fecal Coliform", "Arsenic", "Iron", "Nitrate"]
    vals = [
        values["ph"],
        values["turbidity_ntu"],
        values["bod_mg_l"],
        values["fecal_coliform_mpn"],
        values["arsenic_ug_l"],
        values["iron_mg_l"],
        values["nitrate_mg_l"],
    ]
    fig = px.line_polar(
        r=vals,
        theta=labels,
        line_close=True,
        title=t["param_chart"]
    )
    fig.update_traces(fill="toself", line_color="#2563eb", fillcolor="rgba(37,99,235,0.25)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a"),
        polar=dict(
            bgcolor="rgba(255,255,255,0.75)",
            radialaxis=dict(visible=True, color="#334155"),
            angularaxis=dict(color="#334155")
        ),
    )
    return fig

def build_history_trend(history, t):
    if not history:
        return None
    df = pd.DataFrame(history[:25]).copy()
    if "timestamp" not in df.columns or "overall_probability" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["overall_probability_percent"] = df["overall_probability"] * 100

    fig = px.line(
        df,
        x="timestamp",
        y="overall_probability_percent",
        markers=True,
        title=t["trend_chart"],
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        font=dict(color="#0f172a"),
    )
    return fig

def build_batch_pie(summary_df):
    fig = px.pie(
        summary_df,
        names="Category",
        values="Count",
        color="Category",
        color_discrete_map={
            "High Risk": "#ef4444",
            "Moderate Risk": "#f59e0b",
            "Low Risk": "#10b981",
        },
        hole=0.55,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a"),
    )
    return fig

# =========================================================
# SPLASH SCREEN
# =========================================================
def show_splash_once(t):
    if "splash_seen" not in st.session_state:
        st.session_state.splash_seen = False

    if not st.session_state.splash_seen:
        box = st.empty()
        box.markdown(
            f"""
            <div class="splash-wrap">
                <div class="splash-title">{t["title"]}</div>
                <div class="splash-sub">{t["tagline"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress = st.progress(0)
        status = st.empty()

        for i in range(1, 101, 10):
            progress.progress(i)
            if i < 30:
                status.info(t["loader_1"])
            elif i < 60:
                status.info(t["loader_2"])
            elif i < 90:
                status.info(t["loader_3"])
            else:
                status.success(t["loader_4"])
            time.sleep(0.08)

        st.session_state.splash_seen = True
        box.empty()
        progress.empty()
        status.empty()

# =========================================================
# PAGES
# =========================================================
def show_missing_models(t):
    render_header(t)
    st.error(t["models_missing"])

def show_dashboard(assets, t):
    render_header(t)
    history = load_history()
    metadata = assets["metadata"]

    overall_acc = 0
    for row in metadata.get("results", []):
        if row.get("Disease", "").lower() == "overall":
            overall_acc = row.get("Accuracy", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        animated_counter(f"📋 {t['total_tests']}", len(history))
    with c2:
        animated_counter(f"🚨 {t['alerts']}", sum(1 for x in history if x.get("overall_probability", 0) >= 0.70))
    with c3:
        animated_counter(f"🎯 {t['accuracy']}", f"{overall_acc*100:.1f}", "%")
    with c4:
        animated_counter(f"🧠 {t['features']}", len(metadata.get("features", [])))

    st.markdown(f'<div class="section-title">{t["geo_card_title"]}</div>', unsafe_allow_html=True)
    advisory_card(t["geo_card_title"], t["geo_card_text"], "green")

    st.markdown(f'<div class="section-title">{t["history_title"]}</div>', unsafe_allow_html=True)
    if history:
        df = pd.DataFrame(history[:10]).copy()
        cols = [c for c in ["timestamp", "ui_state", "district", "water_source", "season", "overall_probability", "risk_label"] if c in df.columns]
        if cols:
            show_df = df[cols].copy()
            if "overall_probability" in show_df.columns:
                show_df["overall_probability"] = show_df["overall_probability"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        trend = build_history_trend(history, t)
        if trend:
            st.plotly_chart(trend, use_container_width=True)
    else:
        st.info(t["no_history"])

def show_water_test(assets, t):
    render_header(t)
    st.markdown(f'<div class="section-title">{t["test"]}</div>', unsafe_allow_html=True)
    st.write(t["fill"])

    encoders = assets["encoders"]
    features = assets["features"]
    models = assets["models"]
    scalers = assets["scalers"]

    supported_model_states = get_classes(encoders, "state")
    location_options = get_classes(encoders, "location_type")
    source_options = get_classes(encoders, "water_source")
    season_options = get_classes(encoders, "season")

    with st.form("water_test_form"):
        st.markdown(f"### {t['location']}")
        top1, top2, top3, top4 = st.columns(4)

        with top1:
            ui_state = st.selectbox(
                t["state_ui"],
                INDIA_STATES_UTS,
                index=INDIA_STATES_UTS.index("Maharashtra")
            )

        with top2:
            if ui_state == "Maharashtra":
                district = st.selectbox(t["district_ui"], MAHARASHTRA_DISTRICTS)
            else:
                district = st.text_input(t["district_ui"], placeholder="Enter district / city")

        with top3:
            if ui_state in supported_model_states:
                model_state = ui_state
                st.selectbox(
                    t["model_state"],
                    supported_model_states,
                    index=supported_model_states.index(model_state),
                    disabled=True
                )
            else:
                model_state = st.selectbox(t["model_state"], supported_model_states, index=0)

        with top4:
            season = st.selectbox("Season", season_options)

        mid1, mid2, mid3 = st.columns([1.3, 1, 1])
        with mid1:
            location_type = st.selectbox("Location Type", location_options)
        with mid2:
            water_source = st.selectbox("Water Source", source_options)
        with mid3:
            st.markdown("#####")
            if ui_state == "Maharashtra":
                st.success(t["mh_note"])
            elif ui_state not in supported_model_states:
                st.warning(t["unsupported_state_note"])

        st.caption(f"{t['supported_state_info']}: {', '.join(supported_model_states)}")

        st.markdown(f"### {t['params']}")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            tds_mg_l = st.number_input("TDS (mg/L)", 0.0, 5000.0, 250.0, 1.0)
            bod_mg_l = st.number_input("BOD (mg/L)", 0.0, 100.0, 3.0, 0.1)
            fluoride_mg_l = st.number_input("Fluoride (mg/L)", 0.0, 10.0, 0.8, 0.1)

        with c2:
            turbidity_ntu = st.number_input("Turbidity (NTU)", 0.0, 1000.0, 3.0, 0.1)
            dissolved_oxygen_mg_l = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 6.0, 0.1)
            fecal_coliform_mpn = st.number_input("Fecal Coliform (MPN)", 0.0, 100000.0, 20.0, 1.0)
            chloride_mg_l = st.number_input("Chloride (mg/L)", 0.0, 5000.0, 150.0, 1.0)

        with c3:
            total_coliform_mpn = st.number_input("Total Coliform (MPN)", 0.0, 100000.0, 50.0, 1.0)
            nitrate_mg_l = st.number_input("Nitrate (mg/L)", 0.0, 500.0, 10.0, 0.1)
            hardness_mg_l = st.number_input("Hardness (mg/L)", 0.0, 5000.0, 180.0, 1.0)
            temperature_c = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, 0.1)

        with c4:
            arsenic_ug_l = st.number_input("Arsenic (µg/L)", 0.0, 1000.0, 5.0, 0.1)
            iron_mg_l = st.number_input("Iron (mg/L)", 0.0, 100.0, 0.3, 0.1)
            population_served = st.number_input("Population Served", 0.0, 10000000.0, 1500.0, 1.0)
            sanitation_access_percent = st.slider("Sanitation Access (%)", 0.0, 100.0, 70.0, 1.0)

        submit = st.form_submit_button(t["predict"], use_container_width=True)

    if submit:
        try:
            input_encoded = {
                "state_encoded": safe_transform(encoders, "state", model_state),
                "location_type_encoded": safe_transform(encoders, "location_type", location_type),
                "water_source_encoded": safe_transform(encoders, "water_source", water_source),
                "season_encoded": safe_transform(encoders, "season", season),
            }

            test_params = {
                "ph": ph,
                "turbidity_ntu": turbidity_ntu,
                "tds_mg_l": tds_mg_l,
                "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
                "bod_mg_l": bod_mg_l,
                "fecal_coliform_mpn": fecal_coliform_mpn,
                "total_coliform_mpn": total_coliform_mpn,
                "nitrate_mg_l": nitrate_mg_l,
                "fluoride_mg_l": fluoride_mg_l,
                "chloride_mg_l": chloride_mg_l,
                "hardness_mg_l": hardness_mg_l,
                "temperature_c": temperature_c,
                "arsenic_ug_l": arsenic_ug_l,
                "iron_mg_l": iron_mg_l,
                "population_served": population_served,
                "sanitation_access_percent": sanitation_access_percent,
            }

            input_df = prepare_input_dataframe({**input_encoded, **test_params}, features)
            results = predict_all(input_df, models, scalers)

            overall_prob = results["overall"]["probability"]
            risk_text, risk_level = get_risk(overall_prob)
            safety_score = calculate_water_safety_score(test_params, overall_prob)

            st.markdown("---")
            st.markdown(risk_box(risk_text, risk_level), unsafe_allow_html=True)

            st.markdown(f'<div class="section-title">{t["single_dashboard"]}</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            disease_order = [
                ("🦠 Cholera", "cholera"),
                ("🤒 Typhoid", "typhoid"),
                ("🧫 Dysentery", "dysentery"),
                ("💉 Hepatitis A", "hepatitis_a"),
                ("🚨 Overall", "overall"),
            ]
            for metric, (label, key) in zip([m1, m2, m3, m4, m5], disease_order):
                metric.metric(label, f"{results[key]['probability']*100:.1f}%", "Alert" if results[key]["prediction"] == 1 else "Normal")

            score_col, traffic_col = st.columns(2)
            with score_col:
                st.markdown(
                    f"""
                    <div class="score-box">
                        <div style="font-size:1rem;font-weight:800;color:#1d4ed8;margin-bottom:0.35rem;">💯 {t["water_score"]}</div>
                        <div style="font-size:2.2rem;font-weight:900;color:#0f172a;">{safety_score}/100</div>
                        <div style="font-size:0.92rem;color:#475569;">Higher score means safer overall water condition.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with traffic_col:
                traffic_light_widget(risk_level, t["traffic_title"])

            rows = []
            for label, key in disease_order:
                rows.append({
                    "Disease": label,
                    "Prediction": "Outbreak Alert" if results[key]["prediction"] == 1 else "Normal",
                    "Probability": f"{results[key]['probability']*100:.2f}%"
                })
            st.markdown(f"### {t['results']}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            warnings_list = parameter_warnings(test_params)
            recs = generate_recommendations(results, warnings_list)

            left, right = st.columns([1, 1])
            with left:
                if warnings_list:
                    st.error(f"⚠️ {t['unsafe_msg']}")
                    for w in warnings_list:
                        st.write(f"- {w}")
                else:
                    st.success(f"✅ {t['safe_msg']}")

                st.markdown(f"### {t['recommend']}")
                for r in recs:
                    st.write(f"- {r}")

            with right:
                st.plotly_chart(build_gauge_chart(overall_prob, t), use_container_width=True)

            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                st.plotly_chart(build_probability_chart(results, t), use_container_width=True)
            with bottom_right:
                st.plotly_chart(build_parameter_chart(test_params, t), use_container_width=True)

            st.markdown(f"### {t['health_advisory']}")
            if overall_prob >= 0.70:
                advisory_card("🚨 Critical Health Advisory", "Water shows high outbreak risk. Use boiled or treated water only and inform local health authorities.", "red")
            elif overall_prob >= 0.35:
                advisory_card("⚠️ Preventive Advisory", "Risk is moderate. Retest water, improve sanitation, and avoid untreated direct consumption.", "yellow")
            else:
                advisory_card("✅ Safer Advisory", "Current sample appears comparatively safer, but regular monitoring should continue.", "green")

            if results["cholera"]["probability"] >= 0.60:
                advisory_card("🦠 Cholera Watch", "Ensure chlorination, safe storage, and hygiene precautions.", "red")
            if results["typhoid"]["probability"] >= 0.60:
                advisory_card("🤒 Typhoid Watch", "Avoid unsafe source mixing and prefer filtered drinking water.", "yellow")
            if results["hepatitis_a"]["probability"] >= 0.60:
                advisory_card("💉 Hepatitis A Watch", "Strengthen sanitation awareness and hygiene measures.", "yellow")

            pdf_inputs = {
                "UI State": ui_state,
                "District": district if district else "-",
                "Model State": model_state,
                "Location Type": location_type,
                "Water Source": water_source,
                "Season": season,
                "pH": ph,
                "Turbidity NTU": turbidity_ntu,
                "TDS mg/L": tds_mg_l,
                "DO mg/L": dissolved_oxygen_mg_l,
                "BOD mg/L": bod_mg_l,
                "Fecal Coliform": fecal_coliform_mpn,
                "Total Coliform": total_coliform_mpn,
                "Nitrate mg/L": nitrate_mg_l,
                "Fluoride mg/L": fluoride_mg_l,
                "Chloride mg/L": chloride_mg_l,
                "Hardness mg/L": hardness_mg_l,
                "Temperature C": temperature_c,
                "Arsenic ug/L": arsenic_ug_l,
                "Iron mg/L": iron_mg_l,
                "Population Served": population_served,
                "Sanitation Access %": sanitation_access_percent,
            }

            pdf_buffer = build_pdf_report(pdf_inputs, results, warnings_list, recs, safety_score)
            st.download_button(
                t["export_pdf"],
                data=pdf_buffer,
                file_name=f"water_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ui_state": ui_state,
                "district": district if district else "",
                "model_state": model_state,
                "location_type": location_type,
                "water_source": water_source,
                "season": season,
                "overall_probability": overall_prob,
                "risk_label": risk_text,
                "water_safety_score": safety_score,
                "cholera_probability": results["cholera"]["probability"],
                "typhoid_probability": results["typhoid"]["probability"],
                "dysentery_probability": results["dysentery"]["probability"],
                "hepatitis_a_probability": results["hepatitis_a"]["probability"],
            }
            save_history(record)

        except Exception as e:
            st.error(f"{t['prediction_failed']}: {e}")

def preprocess_batch(df, assets):
    encoders = assets["encoders"]
    features = assets["features"]

    required = [
        "state", "location_type", "water_source", "season",
        "ph", "turbidity_ntu", "tds_mg_l", "dissolved_oxygen_mg_l", "bod_mg_l",
        "fecal_coliform_mpn", "total_coliform_mpn", "nitrate_mg_l", "fluoride_mg_l",
        "chloride_mg_l", "hardness_mg_l", "temperature_c", "arsenic_ug_l",
        "iron_mg_l", "population_served", "sanitation_access_percent"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    for col in ["state", "location_type", "water_source", "season"]:
        mapping = {cls: idx for idx, cls in enumerate(encoders[col].classes_)}
        if not out[col].isin(mapping.keys()).all():
            bad = out.loc[~out[col].isin(mapping.keys()), col].unique().tolist()
            raise ValueError(f"Unknown values in {col}: {bad}")
        out[f"{col}_encoded"] = out[col].map(mapping)

    prepared = out.reindex(columns=features, fill_value=0)
    return prepared, out

def show_batch_analysis(assets, t):
    render_header(t)
    st.markdown(f'<div class="section-title">{t["batch"]}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(t["upload_csv"], type=["csv"])

    with st.expander(t["required_cols"]):
        st.code(
            "state,location_type,water_source,season,ph,turbidity_ntu,tds_mg_l,"
            "dissolved_oxygen_mg_l,bod_mg_l,fecal_coliform_mpn,total_coliform_mpn,"
            "nitrate_mg_l,fluoride_mg_l,chloride_mg_l,hardness_mg_l,temperature_c,"
            "arsenic_ug_l,iron_mg_l,population_served,sanitation_access_percent"
        )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(), use_container_width=True)

            if st.button(t["run_batch"], use_container_width=True):
                prepared, original = preprocess_batch(df, assets)
                models = assets["models"]
                scalers = assets["scalers"]

                results_df = original.copy()

                for disease, model in models.items():
                    scaler = scalers[disease]
                    scaled = scaler.transform(prepared)
                    preds = model.predict(scaled)
                    probs = model.predict_proba(scaled)[:, 1] if hasattr(model, "predict_proba") else preds
                    results_df[f"{disease}_prediction"] = preds
                    results_df[f"{disease}_probability"] = probs

                results_df["overall_risk_label"] = results_df["overall_probability"].apply(lambda x: get_risk(x)[0])

                st.dataframe(results_df, use_container_width=True)

                summary_df = pd.DataFrame({
                    "Category": ["High Risk", "Moderate Risk", "Low Risk"],
                    "Count": [
                        int((results_df["overall_probability"] >= 0.70).sum()),
                        int(((results_df["overall_probability"] >= 0.35) & (results_df["overall_probability"] < 0.70)).sum()),
                        int((results_df["overall_probability"] < 0.35).sum()),
                    ]
                })

                st.markdown(f'<div class="section-title">{t["batch_summary"]}</div>', unsafe_allow_html=True)
                st.plotly_chart(build_batch_pie(summary_df), use_container_width=True)

                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    t["download_csv"],
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"{t['batch_failed']}: {e}")

def show_live_analytics(assets, t):
    render_header(t)
    history = load_history()
    if not history:
        st.info(t["no_history"])
        return

    df = pd.DataFrame(history)
    avg_risk = df["overall_probability"].mean() * 100 if "overall_probability" in df.columns else 0
    high_alerts = int((df["overall_probability"] >= 0.70).sum()) if "overall_probability" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        animated_counter("📊 Avg Overall Risk", f"{avg_risk:.1f}", "%")
    with c2:
        animated_counter("🚨 High Alerts", high_alerts)
    with c3:
        animated_counter("🧪 Records", len(df))

    if "ui_state" in df.columns:
        st.markdown('<div class="section-title">🌍 State-wise Test Count</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="ui_state", color="ui_state", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            font=dict(color="#0f172a"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    if "season" in df.columns and "overall_probability" in df.columns:
        st.markdown('<div class="section-title">🌦️ Season vs Average Risk</div>', unsafe_allow_html=True)
        season_df = df.groupby("season", as_index=False)["overall_probability"].mean()
        season_df["overall_probability"] *= 100
        sfig = px.bar(
            season_df,
            x="season",
            y="overall_probability",
            color="season",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        sfig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            font=dict(color="#0f172a"),
            showlegend=False,
        )
        st.plotly_chart(sfig, use_container_width=True)

    trend = build_history_trend(history, t)
    if trend:
        st.plotly_chart(trend, use_container_width=True)

def show_history(t):
    render_header(t)
    history = load_history()
    if not history:
        st.info(t["no_history"])
        return

    df = pd.DataFrame(history)
    c1, c2, c3 = st.columns(3)

    with c1:
        sel_state = st.selectbox("Filter by State", ["All"] + sorted(df["ui_state"].dropna().unique().tolist())) if "ui_state" in df.columns else "All"
    with c2:
        sel_season = st.selectbox("Filter by Season", ["All"] + sorted(df["season"].dropna().unique().tolist())) if "season" in df.columns else "All"
    with c3:
        sel_risk = st.selectbox("Filter by Risk", ["All"] + sorted(df["risk_label"].dropna().unique().tolist())) if "risk_label" in df.columns else "All"

    filtered = df.copy()
    if sel_state != "All":
        filtered = filtered[filtered["ui_state"] == sel_state]
    if sel_season != "All":
        filtered = filtered[filtered["season"] == sel_season]
    if sel_risk != "All":
        filtered = filtered[filtered["risk_label"] == sel_risk]

    if "overall_probability" in filtered.columns:
        filtered["overall_probability"] = filtered["overall_probability"].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if st.button(t["clear_history"]):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        st.success("History cleared successfully.")
        st.rerun()

def show_awareness_bot(t):
    render_header(t)
    st.markdown(f'<div class="section-title">{t["bot_title"]}</div>', unsafe_allow_html=True)
    st.write(t["bot_desc"])

    left, right = st.columns([2.2, 1])

    with left:
        try:
            components.iframe(
                "https://water-bot-nine.vercel.app/",
                height=720,
                scrolling=True
            )
        except Exception:
            st.warning(t["bot_embed_note"])

    with right:
        advisory_card(
            "💧 Bot Purpose",
            t["bot_desc"],
            "green"
        )
        advisory_card(
            "❓ Suggested Questions",
            f'{t["bot_tip_1"]}\n\n{t["bot_tip_2"]}\n\n{t["bot_tip_3"]}',
            "yellow"
        )

        st.link_button(
            t["bot_open"],
            "https://water-bot-nine.vercel.app/",
            use_container_width=True
        )

        st.info(t["bot_embed_note"])

def show_about(assets, t):
    render_header(t)
    st.markdown('<div class="section-title">About AquaShield India Premium</div>', unsafe_allow_html=True)
    st.write(
        """
        AquaShield India Premium is an AI-based water disease prediction and monitoring dashboard.
        It combines disease prediction, colorful analytics, gauge-based risk visualization,
        health advisory cards, PDF reports, batch analysis, multilingual interface, integrated
        awareness chatbot, and India-ready state selection with Maharashtra district support.
        """
    )
    st.json(assets["metadata"])

def show_deploy_guide(t):
    render_header(t)
    st.markdown(
        f"""
        <div class="deploy-box">
            <h3 style="margin-top:0;">🚀 Deployment Guide</h3>
            <p>{t["deploy_text"]}</p>
            <p><b>Steps:</b></p>
            <p>1. Create a GitHub repository</p>
            <p>2. Upload app.py, models folder, requirements.txt</p>
            <p>3. Open Streamlit Community Cloud</p>
            <p>4. Connect GitHub repo</p>
            <p>5. Select app.py and deploy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("pip install -r requirements.txt\nstreamlit run app.py")

# =========================================================
# MAIN
# =========================================================
def main():
    with st.sidebar:
        lang = st.selectbox("🌐 Language / भाषा / Language", ["English", "Marathi", "Hindi"])
        t = TEXTS[lang]

    show_splash_once(t)
    assets = load_assets()

    with st.sidebar:
        st.markdown(f"## {t['navigation']}")
        page = st.radio(
            "",
            [
                t["dashboard"],
                t["test"],
                t["batch"],
                t["analytics"],
                t["history"],
                t["bot"],
                t["about"],
                t["deploy"],
            ],
        )

        st.markdown("---")
        st.markdown(f"### 📊 {t['stats']}")
        history = load_history()
        st.metric(t["total_tests"], len(history))
        st.metric(t["alerts"], sum(1 for x in history if x.get("overall_probability", 0) >= 0.70))

        if assets and assets.get("metadata", {}).get("results"):
            overall_acc = 0
            for row in assets["metadata"]["results"]:
                if row.get("Disease", "").lower() == "overall":
                    overall_acc = row.get("Accuracy", 0)
                    break
            st.metric(t["accuracy"], f"{overall_acc*100:.1f}%")

        st.markdown("---")
        st.markdown(
            '<div class="small-note">Premium UI • Bot Added • Traffic Light • Water Safety Score • Maharashtra Districts • Gauge Risk • PDF Export</div>',
            unsafe_allow_html=True,
        )

    if assets is None:
        show_missing_models(t)
        return

    if page == t["dashboard"]:
        show_dashboard(assets, t)
    elif page == t["test"]:
        show_water_test(assets, t)
    elif page == t["batch"]:
        show_batch_analysis(assets, t)
    elif page == t["analytics"]:
        show_live_analytics(assets, t)
    elif page == t["history"]:
        show_history(t)
    elif page == t["bot"]:
        show_awareness_bot(t)
    elif page == t["about"]:
        show_about(assets, t)
    elif page == t["deploy"]:
        show_deploy_guide(t)

if __name__ == "__main__":
    main()