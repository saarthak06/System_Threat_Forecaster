"""
System Threat Forecaster — Streamlit App
-----------------------------------------
Matches the notebook pipeline exactly:
  1. Numeric-only features
  2. Median imputation
  3. Remove zero-variance columns
  4. StandardScaler
  5. LightGBMClassifier (final model)

Since `train_small.csv` is not bundled, the app generates realistic
synthetic network/system data and trains at startup (fast — < 2 s).
Drop your own CSV in the same folder and point DATA_PATH to it.
"""

import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib, io, os

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# CONFIG
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
DATA_PATH = "train_small.csv"   # change if you have the real file
SEED      = 42
MODEL_CACHE = "model_cache.joblib"

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FEATURE DEFINITIONS  (network / system-security domain)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
FEATURE_META = {
    # name                  : (min,   max,   step,   default, description)
    "duration"              : (0,     60000, 1,      0,       "Connection duration (seconds)"),
    "protocol_type"         : (0,     2,     1,      1,       "Protocol  0=icmp  1=tcp  2=udp"),
    "src_bytes"             : (0,     100000,100,    491,     "Data bytes from source"),
    "dst_bytes"             : (0,     100000,100,    0,       "Data bytes to destination"),
    "land"                  : (0,     1,     1,      0,       "Same src/dst host+port (1=yes)"),
    "wrong_fragment"        : (0,     3,     1,      0,       "Number of wrong fragments"),
    "urgent"                : (0,     3,     1,      0,       "Number of urgent packets"),
    "hot"                   : (0,     30,    1,      0,       "Hot indicators count"),
    "num_failed_logins"     : (0,     5,     1,      0,       "Failed login attempts"),
    "logged_in"             : (0,     1,     1,      1,       "Successfully logged in (1=yes)"),
    "num_compromised"       : (0,     10,    1,      0,       "Compromised conditions"),
    "root_shell"            : (0,     1,     1,      0,       "Root shell obtained (1=yes)"),
    "su_attempted"          : (0,     2,     1,      0,       "su-root attempt (0/1/2)"),
    "num_root"              : (0,     10,    1,      0,       "Root accesses count"),
    "num_file_creations"    : (0,     10,    1,      0,       "File creation operations"),
    "num_shells"            : (0,     5,     1,      0,       "Shell prompts"),
    "num_access_files"      : (0,     10,    1,      0,       "Access to ctrl files"),
    "count"                 : (0,     512,   1,      2,       "Connections to same host (2 s)"),
    "srv_count"             : (0,     512,   1,      2,       "Connections to same service (2 s)"),
    "serror_rate"           : (0.0,   1.0,   0.01,   0.0,     "SYN error rate"),
    "rerror_rate"           : (0.0,   1.0,   0.01,   0.0,     "REJ error rate"),
    "same_srv_rate"         : (0.0,   1.0,   0.01,   1.0,     "Same service rate"),
    "diff_srv_rate"         : (0.0,   1.0,   0.01,   0.0,     "Different service rate"),
    "dst_host_count"        : (0,     255,   1,      255,     "Dst-host connection count"),
    "dst_host_srv_count"    : (0,     255,   1,      11,      "Dst-host same-service count"),
    "dst_host_same_srv_rate": (0.0,   1.0,   0.01,   0.07,   "Dst-host same-service rate"),
    "dst_host_diff_srv_rate": (0.0,   1.0,   0.01,   0.06,   "Dst-host diff-service rate"),
    "dst_host_serror_rate"  : (0.0,   1.0,   0.01,   0.0,    "Dst-host SYN error rate"),
    "dst_host_rerror_rate"  : (0.0,   1.0,   0.01,   0.0,    "Dst-host REJ error rate"),
}

LABEL_NAMES = {0: "Normal", 1: "Threat Detected"}
LABEL_COLORS = {0: "#2ecc71", 1: "#e74c3c"}


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# DATA GENERATION  (only used when real CSV is absent)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
def generate_synthetic_data(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        threat = rng.integers(0, 2)
        row = {}
        if threat:
            row["duration"]           = rng.integers(0, 5)
            row["protocol_type"]      = rng.choice([0, 2])
            row["src_bytes"]          = int(rng.exponential(30000))
            row["dst_bytes"]          = int(rng.exponential(500))
            row["land"]               = int(rng.random() < 0.1)
            row["wrong_fragment"]     = int(rng.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05]))
            row["urgent"]             = int(rng.choice([0, 1], p=[0.7, 0.3]))
            row["hot"]                = int(rng.integers(0, 20))
            row["num_failed_logins"]  = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))
            row["logged_in"]          = int(rng.random() < 0.3)
            row["num_compromised"]    = int(rng.integers(0, 8))
            row["root_shell"]         = int(rng.random() < 0.3)
            row["su_attempted"]       = int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
            row["num_root"]           = int(rng.integers(0, 8))
            row["num_file_creations"] = int(rng.integers(0, 6))
            row["num_shells"]         = int(rng.integers(0, 4))
            row["num_access_files"]   = int(rng.integers(0, 8))
            row["count"]              = int(rng.integers(1, 512))
            row["srv_count"]          = int(rng.integers(1, 512))
            row["serror_rate"]        = round(float(rng.uniform(0.3, 1.0)), 2)
            row["rerror_rate"]        = round(float(rng.uniform(0.0, 0.8)), 2)
            row["same_srv_rate"]      = round(float(rng.uniform(0.0, 0.5)), 2)
            row["diff_srv_rate"]      = round(float(rng.uniform(0.3, 1.0)), 2)
            row["dst_host_count"]     = int(rng.integers(1, 255))
            row["dst_host_srv_count"] = int(rng.integers(1, 255))
            row["dst_host_same_srv_rate"] = round(float(rng.uniform(0.0, 0.4)), 2)
            row["dst_host_diff_srv_rate"] = round(float(rng.uniform(0.3, 1.0)), 2)
            row["dst_host_serror_rate"]   = round(float(rng.uniform(0.2, 1.0)), 2)
            row["dst_host_rerror_rate"]   = round(float(rng.uniform(0.0, 0.8)), 2)
        else:
            row["duration"]           = int(rng.integers(0, 3000))
            row["protocol_type"]      = 1
            row["src_bytes"]          = int(rng.exponential(3000))
            row["dst_bytes"]          = int(rng.exponential(5000))
            row["land"]               = 0
            row["wrong_fragment"]     = 0
            row["urgent"]             = 0
            row["hot"]                = int(rng.integers(0, 3))
            row["num_failed_logins"]  = 0
            row["logged_in"]          = 1
            row["num_compromised"]    = 0
            row["root_shell"]         = 0
            row["su_attempted"]       = 0
            row["num_root"]           = 0
            row["num_file_creations"] = 0
            row["num_shells"]         = 0
            row["num_access_files"]   = 0
            row["count"]              = int(rng.integers(1, 10))
            row["srv_count"]          = int(rng.integers(1, 10))
            row["serror_rate"]        = 0.0
            row["rerror_rate"]        = 0.0
            row["same_srv_rate"]      = round(float(rng.uniform(0.8, 1.0)), 2)
            row["diff_srv_rate"]      = 0.0
            row["dst_host_count"]     = int(rng.integers(50, 255))
            row["dst_host_srv_count"] = int(rng.integers(10, 255))
            row["dst_host_same_srv_rate"] = round(float(rng.uniform(0.6, 1.0)), 2)
            row["dst_host_diff_srv_rate"] = round(float(rng.uniform(0.0, 0.2)), 2)
            row["dst_host_serror_rate"]   = 0.0
            row["dst_host_rerror_rate"]   = 0.0
        row["label"] = threat
        rows.append(row)
    return pd.DataFrame(rows)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# TRAINING PIPELINE  (mirrors the notebook exactly)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
@st.cache_resource(show_spinner="Training model — this takes only a moment…")
def train_model():
    # Load or generate data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        source = f"Loaded `{DATA_PATH}`"
    else:
        df = generate_synthetic_data()
        source = "Using synthetic data (place `train_small.csv` here to use real data)"

    # Notebook preprocessing steps ↓
    df = df.fillna(df.median(numeric_only=True))          # median imputation
    df = df.select_dtypes(include=["number"])             # numeric only
    df = df.loc[:, df.nunique() > 1]                      # drop zero-variance

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED
    )

    model = LGBMClassifier(n_estimators=200, max_depth=7, random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)

    preds   = model.predict(X_val)
    acc     = accuracy_score(y_val, preds)
    cm      = confusion_matrix(y_val, preds)
    classes = sorted(y.unique())

    return model, scaler, feature_names, acc, cm, classes, source


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# PAGE SETUP
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
st.set_page_config(
    page_title="System Threat Forecaster",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ System Threat Forecaster")
st.caption("Predict whether a network connection is **Normal** or a **Threat** using LightGBM.")

model, scaler, feature_names, val_acc, cm, classes, data_source = train_model()

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# SIDEBAR — model info
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Validation Accuracy", f"{val_acc * 100:.2f}%")
    st.caption(data_source)

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {LABEL_NAMES.get(c, c)}" for c in classes],
        columns=[f"Pred: {LABEL_NAMES.get(c, c)}" for c in classes],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Features Used")
    st.write(", ".join(feature_names))

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# MAIN — input widgets
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
st.markdown("### 🔧 Connection Features")
st.info("Adjust the sliders / inputs below, then click **Predict**.")

input_values = {}

# Split features into 3 columns for a clean layout
cols = st.columns(3)
for idx, feat in enumerate(feature_names):
    col = cols[idx % 3]

    if feat in FEATURE_META:
        lo, hi, step, default, desc = FEATURE_META[feat]
    else:
        lo, hi, step, default, desc = 0, 1000, 1, 0, feat.replace("_", " ").title()

    with col:
        is_float = isinstance(step, float)
        if is_float:
            val = st.slider(
                feat,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                help=desc,
            )
        elif hi - lo <= 10:
            val = st.number_input(
                feat,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default),
                step=int(step),
                help=desc,
            )
        else:
            val = st.slider(
                feat,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default),
                step=int(step),
                help=desc,
            )
        input_values[feat] = val

st.markdown("---")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# PREDICT
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([input_values])[feature_names]

    # Fill any missing features with 0 (safety net)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    X_input = scaler.transform(input_df)

    prediction      = model.predict(X_input)[0]
    probabilities   = model.predict_proba(X_input)[0]

    label  = LABEL_NAMES.get(int(prediction), str(prediction))
    color  = LABEL_COLORS.get(int(prediction), "#888")
    conf   = probabilities[int(prediction)] * 100

    st.markdown("## 🎯 Prediction Result")

    result_col, prob_col = st.columns([1, 2])

    with result_col:
        st.markdown(
            f"""
            <div style="background:{color};padding:28px 20px;border-radius:12px;text-align:center;">
                <span style="font-size:2.2rem;font-weight:700;color:white;">{label}</span><br>
                <span style="color:rgba(255,255,255,.85);font-size:1rem;">Confidence: {conf:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown("**Class Probabilities**")
        for cls, prob in zip(classes, probabilities):
            cls_name = LABEL_NAMES.get(int(cls), str(cls))
            st.progress(float(prob), text=f"{cls_name}: {prob*100:.1f}%")

    st.markdown("---")
    with st.expander("📋 Feature values sent to model"):
        display = pd.DataFrame(
            {"Feature": feature_names, "Value": [input_values[f] for f in feature_names]}
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
