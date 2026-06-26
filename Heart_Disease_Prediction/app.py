

import os
import pickle
import numpy as np
import streamlit as st

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "meta.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #1a0000, #7f0000, #b71c1c);
        padding: 2rem; border-radius: 16px;
        text-align: center; margin-bottom: 1.5rem;
    }
    .header h1 { color: #fff5f5; font-size: 2.2rem; margin: 0; }
    .header p  { color: #ffcdd2; margin: 0.4rem 0 0; font-size: 1rem; }

    .sec-title {
        font-size: 1.05rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #ef4444;
        padding-left: 10px; margin: 1.6rem 0 0.8rem;
    }

    .card-safe {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981; border-radius: 16px;
        padding: 1.8rem; text-align: center; margin-top: 1rem;
    }
    .card-safe h2 { color: #064e3b; margin: 0; font-size: 1.8rem; }
    .card-safe p  { color: #065f46; margin: 0.5rem 0 0; font-size: 0.95rem; }

    .card-risk {
        background: linear-gradient(135deg, #fee2e2, #fca5a5);
        border: 2px solid #ef4444; border-radius: 16px;
        padding: 1.8rem; text-align: center; margin-top: 1rem;
    }
    .card-risk h2 { color: #7f1d1d; margin: 0; font-size: 1.8rem; }
    .card-risk p  { color: #991b1b; margin: 0.5rem 0 0; font-size: 0.95rem; }

    .tip-box {
        background: #fff7ed; border: 1px solid #fed7aa;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-top: 1rem; font-size: 0.9rem;
        color: #7c2d12; line-height: 1.8;
    }
    .tip-box b { color: #c2410c; display: block; margin-bottom: 0.3rem; }

    .risk-bar-wrap {
        background: #e2e8f0; border-radius: 999px;
        height: 16px; margin: 4px 0; overflow: hidden;
    }
    .risk-bar {
        height: 100%; border-radius: 999px;
    }
    .footer {
        text-align: center; color: #94a3b8;
        font-size: 0.8rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_all():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(META_PATH,   "rb") as f: meta   = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_all()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>❤️ Heart Disease Predictor</h1>
    <p>AI-powered heart disease risk prediction using clinical parameters</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model not found! Run in VS Code terminal:\n\n"
             f"```\ncd {BASE_DIR}\npython train_model.py\n```")
    st.stop()

# ════════════════════════════════════════════════════════════
#  INPUT FORM
# ════════════════════════════════════════════════════════════

# ── Section 1: Personal Info ─────────────────────────────────
st.markdown('<div class="sec-title">👤 Personal Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

age = c1.number_input("Age (years)", 20, 80, 45, 1)
sex = c2.selectbox("Sex", options=[1, 0],
                   format_func=lambda x: "Male" if x == 1 else "Female")
cp  = c3.selectbox("Chest Pain Type",
                   options=[1, 2, 3, 4],
                   format_func=lambda x: {
                       1: "1 — Typical Angina",
                       2: "2 — Atypical Angina",
                       3: "3 — Non-Anginal Pain",
                       4: "4 — Asymptomatic"
                   }[x],
                   index=0,
                   help="Type of chest pain experienced")

# ── Section 2: Vitals ────────────────────────────────────────
st.markdown('<div class="sec-title">🩺 Vital Signs</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)

trestbps = c4.number_input("Resting Blood Pressure (mm Hg)",
                            80, 220, 130, 1,
                            help="Blood pressure at rest")
chol     = c5.number_input("Serum Cholesterol (mg/dl)",
                            100, 600, 240, 5,
                            help="Cholesterol level in blood")

c6, c7 = st.columns(2)
thalach  = c6.number_input("Max Heart Rate Achieved",
                            60, 220, 150, 1,
                            help="Maximum heart rate during exercise test")
oldpeak  = c7.number_input("ST Depression (oldpeak)",
                            0.0, 7.0, 1.0, 0.1,
                            help="ST depression induced by exercise")

# ── Section 3: Medical Tests ──────────────────────────────────
st.markdown('<div class="sec-title">🔬 Medical Test Results</div>', unsafe_allow_html=True)
c8, c9, c10 = st.columns(3)

fbs     = c8.selectbox("Fasting Blood Sugar > 120 mg/dl",
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")
restecg = c9.selectbox("Resting ECG Result",
                        options=[0, 1, 2],
                        format_func=lambda x: {
                            0: "0 — Normal",
                            1: "1 — ST-T Abnormality",
                            2: "2 — Left Ventricular Hypertrophy"
                        }[x])
exang   = c10.selectbox("Exercise Induced Angina",
                         options=[0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No",
                         help="Chest pain during exercise")

c11, c12, c13 = st.columns(3)
slope   = c11.selectbox("Slope of Peak ST Segment",
                         options=[1, 2, 3],
                         format_func=lambda x: {
                             1: "1 — Upsloping",
                             2: "2 — Flat",
                             3: "3 — Downsloping"
                         }[x])
ca      = c12.selectbox("Major Vessels (Fluoroscopy)",
                         options=[0, 1, 2, 3],
                         format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}",
                         help="Number of major vessels colored by fluoroscopy")
thal    = c13.selectbox("Thalassemia",
                         options=[3, 6, 7],
                         format_func=lambda x: {
                             3: "3 — Normal",
                             6: "6 — Fixed Defect",
                             7: "7 — Reversable Defect"
                         }[x])

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════════
if st.button("❤️ Predict Heart Disease Risk", use_container_width=True, type="primary"):

    input_vec    = np.array([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_vec)
    pred         = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    confidence   = proba[pred] * 100
    risk_pct     = proba[1] * 100

    # ── Result card ──────────────────────────────────────────
    if pred == 0:
        st.markdown(f"""
        <div class="card-safe">
            <h2>✅ Low Risk of Heart Disease</h2>
            <p>Confidence: {confidence:.1f}% &nbsp;·&nbsp; Your clinical parameters look healthy!</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <b>💪 Keep your heart healthy:</b>
            • Exercise at least 30 minutes daily<br>
            • Keep cholesterol below 200 mg/dl<br>
            • Maintain blood pressure below 120/80 mm Hg<br>
            • Avoid smoking and excessive alcohol<br>
            • Get annual cardiac checkups after age 40
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card-risk">
            <h2>⚠️ High Risk of Heart Disease</h2>
            <p>Confidence: {confidence:.1f}% &nbsp;·&nbsp; Please consult a cardiologist immediately!</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <b>🚨 Immediate steps to take:</b>
            • Consult a cardiologist as soon as possible<br>
            • Get an ECG and stress test done<br>
            • Monitor blood pressure daily<br>
            • Reduce salt, fried food, and red meat<br>
            • Avoid strenuous physical activity until evaluated<br>
            • Take prescribed medications regularly<br>
            • Quit smoking immediately — it doubles heart risk
        </div>""", unsafe_allow_html=True)

    # ── Risk meter ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📊 Risk Probability**")

    bar_color = "#ef4444" if risk_pct > 60 else "#f59e0b" if risk_pct > 35 else "#10b981"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:8px 0">
        <div style="width:100px;font-size:13px;color:#374151;font-weight:500">Heart Disease</div>
        <div style="flex:1;background:#e2e8f0;border-radius:999px;height:18px;overflow:hidden">
            <div style="width:{risk_pct:.1f}%;height:100%;background:{bar_color};border-radius:999px"></div>
        </div>
        <div style="width:50px;font-size:14px;font-weight:700;color:{bar_color};text-align:right">{risk_pct:.1f}%</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;margin:8px 0">
        <div style="width:100px;font-size:13px;color:#374151;font-weight:500">No Disease</div>
        <div style="flex:1;background:#e2e8f0;border-radius:999px;height:18px;overflow:hidden">
            <div style="width:{proba[0]*100:.1f}%;height:100%;background:#10b981;border-radius:999px"></div>
        </div>
        <div style="width:50px;font-size:14px;font-weight:700;color:#10b981;text-align:right">{proba[0]*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key risk factors ─────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔍 Your Key Risk Factors**")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Age",         age,
              "⚠️ Risk" if age > 55 else "✅ OK")
    f2.metric("Cholesterol", f"{chol} mg/dl",
              "⚠️ High" if chol > 240 else "✅ OK")
    f3.metric("Blood Pressure", f"{trestbps} mmHg",
              "⚠️ High" if trestbps > 140 else "✅ OK")
    f4.metric("Max Heart Rate", thalach,
              "⚠️ Low" if thalach < 120 else "✅ OK")

st.markdown(
    '<div class="footer">⚠️ For educational purposes only. Always consult a doctor for medical advice.</div>',
    unsafe_allow_html=True
)
