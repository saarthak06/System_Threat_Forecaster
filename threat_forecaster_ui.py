"""
System Threat Forecaster — Upload & Predict
--------------------------------------------
Upload your CSV → get instant predictions for every row.
Trained with LightGBM (best model from notebook).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io, os, warnings, time
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
SEED        = 42
LABEL_COL   = "HasDetections"          # Microsoft dataset target column
LABEL_MAP   = {0: "✅ Safe", 1: "⚠️ Threat"}
COLOR_SAFE  = "#3fb950"
COLOR_THREAT= "#f85149"
COLOR_ACCENT= "#58a6ff"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", family="'JetBrains Mono', monospace"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Threat Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  background-color: #0d1117 !important;
  color: #e6edf3 !important;
  font-family: 'Sora', sans-serif;
}
.stApp { background-color: #0d1117; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg,#0d1117 0%,#111823 60%,#0d1117 100%);
  border: 1px solid #21262d;
  border-radius: 16px;
  padding: 36px 40px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero::after {
  content:'';
  position:absolute;top:-80px;right:-80px;
  width:380px;height:380px;
  background:radial-gradient(circle,rgba(88,166,255,.07) 0%,transparent 65%);
  pointer-events:none;
}
.hero-tag {
  font-family:'JetBrains Mono',monospace;
  font-size:11px;font-weight:600;letter-spacing:3px;
  color:#58a6ff;text-transform:uppercase;margin-bottom:10px;
}
.hero-title {
  font-size:2.4rem;font-weight:700;
  background:linear-gradient(90deg,#e6edf3,#58a6ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:0 0 8px;
}
.hero-sub { color:#8b949e;font-size:.95rem;line-height:1.6; }
.badge {
  display:inline-block;
  background:#1c2a3a;border:1px solid #1f6feb;
  border-radius:20px;color:#58a6ff;
  font-family:'JetBrains Mono';font-size:11px;
  padding:3px 12px;margin:8px 5px 0 0;
}

/* ── Feature cards ── */
.feature-card {
  background:#161b22;
  border:1px solid #21262d;
  border-radius:14px;
  padding:28px 30px;
  height:100%;
  transition:border-color .2s;
}
.feature-card:hover { border-color:#30363d; }
.card-num {
  font-family:'JetBrains Mono';font-size:11px;font-weight:700;
  color:#58a6ff;letter-spacing:2px;text-transform:uppercase;
  margin-bottom:8px;
}
.card-title {
  font-size:1.25rem;font-weight:700;color:#e6edf3;
  margin-bottom:6px;
}
.card-desc { color:#8b949e;font-size:.88rem;line-height:1.6; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background:#161b22 !important;
  border:2px dashed #30363d !important;
  border-radius:12px !important;
  padding:10px !important;
  transition:border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color:#58a6ff !important;
}

/* ── Predict button ── */
.stButton > button {
  background:linear-gradient(135deg,#1f6feb,#388bfd) !important;
  color:white !important;
  border:none !important;
  border-radius:10px !important;
  font-family:'JetBrains Mono' !important;
  font-weight:700 !important;
  font-size:15px !important;
  padding:16px 0 !important;
  width:100% !important;
  letter-spacing:.8px !important;
  transition:all .2s !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg,#388bfd,#58a6ff) !important;
  box-shadow:0 8px 32px rgba(88,166,255,.28) !important;
  transform:translateY(-2px) !important;
}
.stButton > button:disabled {
  background:#21262d !important;
  color:#8b949e !important;
  transform:none !important;
  box-shadow:none !important;
}

/* ── Metric tiles ── */
[data-testid="metric-container"] {
  background:#161b22;
  border:1px solid #21262d;
  border-radius:12px;
  padding:18px 22px !important;
}
[data-testid="stMetricValue"] {
  color:#58a6ff !important;
  font-family:'JetBrains Mono' !important;
  font-size:1.8rem !important;
}
[data-testid="stMetricLabel"] { color:#8b949e !important; }

/* ── Result banner ── */
.result-safe {
  background:linear-gradient(135deg,#0d2818,#0f2f1c);
  border:2px solid #3fb950;
  border-radius:14px;padding:22px 28px;
  display:flex;align-items:center;gap:16px;
}
.result-threat {
  background:linear-gradient(135deg,#2d0d0d,#3a1010);
  border:2px solid #f85149;
  border-radius:14px;padding:22px 28px;
  display:flex;align-items:center;gap:16px;
}
.result-icon { font-size:2.4rem; }
.result-label {
  font-family:'JetBrains Mono';font-size:1.5rem;font-weight:700;
}
.result-sub { color:rgba(255,255,255,.65);font-size:.88rem;margin-top:4px; }

/* ── Section title ── */
.section-title {
  font-family:'JetBrains Mono';font-size:11px;font-weight:600;
  color:#58a6ff;letter-spacing:2px;text-transform:uppercase;
  border-bottom:1px solid #21262d;padding-bottom:6px;
  margin:24px 0 14px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border:1px solid #21262d !important;
  border-radius:10px !important;
}

/* ── Tabs ── */
[data-baseweb="tab"] {
  font-family:'JetBrains Mono' !important;
  font-size:12px !important;
  color:#8b949e !important;
  letter-spacing:.5px !important;
}
[aria-selected="true"] {
  color:#58a6ff !important;
  border-bottom:2px solid #58a6ff !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background:#161b22 !important;
  color:#58a6ff !important;
  border:1px solid #30363d !important;
  border-radius:8px !important;
  font-family:'JetBrains Mono' !important;
  font-size:13px !important;
  padding:10px 20px !important;
  width:auto !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color:#58a6ff !important;
  background:#1c2a3a !important;
}

/* ── Alert ── */
.stAlert { border-radius:10px !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
  background:linear-gradient(90deg,#1f6feb,#58a6ff) !important;
  border-radius:4px !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, scaler=None, feature_cols=None, fit=False):
    """Mirror the Colab notebook preprocessing pipeline."""
    df = df.copy()

    # Drop ID / target cols if present
    drop_cols = ["MachineIdentifier", LABEL_COL]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Encode object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Fill NaN with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Align columns if we have a reference set
    if feature_cols is not None:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[feature_cols]

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        return X, scaler, list(df.columns)
    else:
        X = scaler.transform(df)
        return X


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING  (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_trained_model(data_bytes: bytes):
    """Train LightGBM on provided CSV bytes."""
    df = pd.read_csv(io.BytesIO(data_bytes))

    if LABEL_COL not in df.columns:
        # Fallback: assume last column is target
        df.rename(columns={df.columns[-1]: LABEL_COL}, inplace=True)

    y = df[LABEL_COL].copy()
    X_scaled, scaler, feature_cols = preprocess(df, fit=True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        random_state=SEED, verbose=-1
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    acc   = accuracy_score(y_val, preds)
    cm    = confusion_matrix(y_val, preds)
    fpr, tpr, _ = roc_curve(y_val, proba)
    roc_auc     = auc(fpr, tpr)
    importances = model.feature_importances_

    return model, scaler, feature_cols, acc, cm, fpr, tpr, roc_auc, importances


# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════
def _ax(fig):
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e")
    return fig

def pie_chart(safe, threat):
    fig = go.Figure(go.Pie(
        labels=["Safe", "Threat"],
        values=[safe, threat],
        marker=dict(colors=[COLOR_SAFE, COLOR_THREAT]),
        hole=0.62,
        textinfo="percent+label",
        textfont=dict(size=13, family="JetBrains Mono"),
        hovertemplate="%{label}: %{value} records<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{safe+threat}</b><br><span style='font-size:11px'>records</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e6edf3", family="JetBrains Mono"),
    )
    fig.update_layout(**PLOT_LAYOUT, title="Prediction Breakdown", height=300, showlegend=True)
    return fig

def bar_chart(safe, threat):
    fig = go.Figure()
    fig.add_bar(x=["Safe", "Threat"], y=[safe, threat],
                marker_color=[COLOR_SAFE, COLOR_THREAT],
                text=[safe, threat], textposition="outside",
                textfont=dict(family="JetBrains Mono", size=14),
                hovertemplate="%{x}: %{y}<extra></extra>")
    fig.update_layout(**PLOT_LAYOUT, title="Count by Class",
                      height=280, showlegend=False, yaxis_title="Records")
    return _ax(fig)

def confidence_histogram(proba_series):
    fig = go.Figure()
    fig.add_histogram(x=proba_series, nbinsx=40,
                      marker_color=COLOR_ACCENT, opacity=0.8,
                      hovertemplate="Confidence %{x:.2f}: %{y} records<extra></extra>")
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLOR_THREAT,
                  annotation_text="Decision boundary",
                  annotation_font_color=COLOR_THREAT)
    fig.update_layout(**PLOT_LAYOUT, title="Threat Confidence Distribution",
                      xaxis_title="P(Threat)", yaxis_title="Count", height=280)
    return _ax(fig)

def importance_chart(importances, feature_cols, top_n=15):
    idx   = np.argsort(importances)[::-1][:top_n]
    feats = [feature_cols[i] for i in idx]
    vals  = [importances[i]  for i in idx]
    colors = [COLOR_THREAT if v > np.percentile(vals, 70)
              else COLOR_ACCENT if v > np.percentile(vals, 35)
              else "#8b949e" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=feats[::-1], orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, title=f"Top {top_n} Feature Importances",
                      height=max(300, top_n * 28), xaxis_title="LightGBM Importance")
    return _ax(fig)

def confusion_heatmap(cm):
    labels = ["Safe", "Threat"]
    pct = cm.astype(float) / cm.sum() * 100
    text = [[f"{cm[i,j]}<br>{pct[i,j]:.1f}%" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#161b22"],[0.5,"#1a3a5c"],[1,COLOR_ACCENT]],
        showscale=False,
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Confusion Matrix (validation set)", height=260)
    return _ax(fig)

def roc_chart(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines",
                    name=f"LightGBM  AUC={roc_auc:.3f}",
                    line=dict(color=COLOR_ACCENT, width=2.5))
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                    line=dict(color="#8b949e", dash="dash", width=1.5))
    fig.update_layout(**PLOT_LAYOUT, title="ROC Curve (validation set)",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      height=260,
                      legend=dict(x=0.55, y=0.1))
    return _ax(fig)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for key in ["trained", "results_df", "model_meta"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-tag">🛡️ Microsoft Threat Intelligence</div>
  <div class="hero-title">System Threat Forecaster</div>
  <div class="hero-sub">
    Upload your dataset CSV — get instant threat predictions powered by LightGBM.<br>
    Trained on the Microsoft Malware Prediction pipeline from Kaggle.
  </div>
  <span class="badge">LightGBM</span>
  <span class="badge">RandomForest</span>
  <span class="badge">XGBoost</span>
  <span class="badge">Logistic Regression</span>
  <span class="badge">Best Model: LightGBM ✓</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TWO MAIN FEATURE CARDS
# ═══════════════════════════════════════════════════════════════
col_f1, col_f2 = st.columns(2, gap="large")

# ── FEATURE 1 — Upload Training Data ────────────────────────────
with col_f1:
    st.markdown("""
    <div class="feature-card">
      <div class="card-num">Feature 01</div>
      <div class="card-title">📂 Upload Training Data</div>
      <div class="card-desc">Upload your <code>train.csv</code> (with <b>HasDetections</b> column).
      The model will be trained automatically using the LightGBM pipeline.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    train_file = st.file_uploader(
        "Drop train CSV here or click to browse",
        type=["csv"],
        key="train_upload",
        label_visibility="collapsed",
    )

    if train_file:
        st.success(f"✓  `{train_file.name}`  —  {train_file.size / 1024:.1f} KB uploaded")

        if st.button("⚙️  Train LightGBM Model", key="btn_train"):
            with st.spinner("Reading data…"):
                raw_bytes = train_file.read()

            progress = st.progress(0, text="Preprocessing features…")
            time.sleep(0.4); progress.progress(25, "Encoding & scaling…")
            time.sleep(0.3); progress.progress(50, "Training LightGBM…")

            try:
                (model, scaler, feature_cols,
                 acc, cm, fpr, tpr, roc_auc, importances) = get_trained_model(raw_bytes)

                progress.progress(90, "Finalising…")
                time.sleep(0.3)
                progress.progress(100, "Done!")
                time.sleep(0.4)
                progress.empty()

                st.session_state.trained = {
                    "model": model, "scaler": scaler,
                    "feature_cols": feature_cols,
                    "acc": acc, "cm": cm,
                    "fpr": fpr, "tpr": tpr,
                    "roc_auc": roc_auc,
                    "importances": importances,
                }
                st.balloons()
                st.success(f"✅  Model trained!  Accuracy: **{acc*100:.2f}%**  |  AUC: **{roc_auc:.4f}**")

            except Exception as e:
                progress.empty()
                st.error(f"Training failed: {e}")


# ── FEATURE 2 — Upload Prediction File ──────────────────────────
with col_f2:
    st.markdown("""
    <div class="feature-card">
      <div class="card-num">Feature 02</div>
      <div class="card-title">🔍 Upload File & Get Predictions</div>
      <div class="card-desc">Upload any CSV (with or without <b>HasDetections</b>).
      The model runs on every row and returns <b>Safe / Threat</b> predictions with confidence scores.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    pred_file = st.file_uploader(
        "Drop prediction CSV here or click to browse",
        type=["csv"],
        key="pred_upload",
        label_visibility="collapsed",
    )

    if pred_file:
        st.success(f"✓  `{pred_file.name}`  —  {pred_file.size / 1024:.1f} KB uploaded")

    btn_disabled = (st.session_state.trained is None) or (pred_file is None)

    if st.button(
        "🚀  Run Predictions on All Rows",
        key="btn_predict",
        disabled=btn_disabled,
    ):
        if st.session_state.trained is None:
            st.warning("⚠️  Please train the model first (Feature 01).")
        elif pred_file is None:
            st.warning("⚠️  Please upload a prediction file.")
        else:
            with st.spinner("Running predictions…"):
                t = st.session_state.trained
                raw_df = pd.read_csv(pred_file)

                # Keep original for display
                display_df = raw_df.copy()

                X_pred = preprocess(raw_df, scaler=t["scaler"],
                                    feature_cols=t["feature_cols"])
                preds  = t["model"].predict(X_pred)
                proba  = t["model"].predict_proba(X_pred)

                display_df["Prediction"]       = ["Threat" if p == 1 else "Safe" for p in preds]
                display_df["Threat_Prob_%"]    = (proba[:, 1] * 100).round(2)
                display_df["Safe_Prob_%"]      = (proba[:, 0] * 100).round(2)
                display_df["Confidence_%"]     = (np.max(proba, axis=1) * 100).round(2)

                st.session_state.results_df = display_df

    if st.session_state.trained is None:
        st.caption("🔒  Train a model first to unlock predictions.")


# ═══════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════
if st.session_state.results_df is not None:
    df_res = st.session_state.results_df
    n_threat = int((df_res["Prediction"] == "Threat").sum())
    n_safe   = int((df_res["Prediction"] == "Safe").sum())
    n_total  = len(df_res)
    threat_pct = n_threat / n_total * 100

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Prediction Results</p>', unsafe_allow_html=True)

    # ── Summary banner ──────────────────────────────────────────
    if threat_pct >= 50:
        st.markdown(f"""
        <div class="result-threat">
          <div class="result-icon">⚠️</div>
          <div>
            <div class="result-label" style="color:{COLOR_THREAT}">HIGH THREAT DETECTED</div>
            <div class="result-sub">{n_threat} of {n_total} records flagged as threats ({threat_pct:.1f}%)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
          <div class="result-icon">✅</div>
          <div>
            <div class="result-label" style="color:{COLOR_SAFE}">MOSTLY SAFE</div>
            <div class="result-sub">{n_safe} of {n_total} records are safe — {n_threat} threat(s) detected ({threat_pct:.1f}%)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── KPI metrics ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{n_total:,}")
    k2.metric("✅ Safe",        f"{n_safe:,}",   delta=f"{100-threat_pct:.1f}%")
    k3.metric("⚠️ Threats",     f"{n_threat:,}", delta=f"{threat_pct:.1f}%",
              delta_color="inverse")
    k4.metric("Avg Confidence",
              f"{df_res['Confidence_%'].mean():.1f}%")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.plotly_chart(pie_chart(n_safe, n_threat),
                        use_container_width=True, config={"displayModeBar": False})
    with ch2:
        st.plotly_chart(bar_chart(n_safe, n_threat),
                        use_container_width=True, config={"displayModeBar": False})
    with ch3:
        st.plotly_chart(confidence_histogram(df_res["Threat_Prob_%"] / 100),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Tabbed results ──────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Detailed Results</p>', unsafe_allow_html=True)

    tab_all, tab_threat, tab_safe, tab_model = st.tabs([
        f"📄 All Records ({n_total})",
        f"⚠️ Threats ({n_threat})",
        f"✅ Safe ({n_safe})",
        "🤖 Model Analytics",
    ])

    # colour-coded prediction column
    def style_df(df):
        def color_row(row):
            c = "background-color:#2d0d0d" if row["Prediction"] == "Threat" else "background-color:#0d2818"
            return [c] * len(row)
        return df.style.apply(color_row, axis=1)

    SHOW_COLS = ["Prediction", "Threat_Prob_%", "Safe_Prob_%", "Confidence_%"] + \
                [c for c in df_res.columns if c not in
                 ["Prediction","Threat_Prob_%","Safe_Prob_%","Confidence_%",LABEL_COL]]

    with tab_all:
        st.dataframe(
            df_res[SHOW_COLS].reset_index(drop=True),
            use_container_width=True,
            height=420,
            column_config={
                "Prediction"    : st.column_config.TextColumn("Prediction", width="small"),
                "Threat_Prob_%": st.column_config.ProgressColumn(
                    "Threat %", min_value=0, max_value=100, format="%.1f%%"),
                "Confidence_%"  : st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

        # Download
        csv_out = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download Full Results CSV",
            data=csv_out,
            file_name="threat_predictions.csv",
            mime="text/csv",
        )

    with tab_threat:
        threats = df_res[df_res["Prediction"] == "Threat"][SHOW_COLS].reset_index(drop=True)
        if len(threats):
            st.dataframe(threats, use_container_width=True, height=380,
                         column_config={
                             "Threat_Prob_%": st.column_config.ProgressColumn(
                                 "Threat %", min_value=0, max_value=100, format="%.1f%%"),
                             "Confidence_%": st.column_config.ProgressColumn(
                                 "Confidence", min_value=0, max_value=100, format="%.1f%%"),
                         })
            st.download_button(
                "⬇️  Download Threats Only",
                data=threats.to_csv(index=False).encode("utf-8"),
                file_name="threats_only.csv", mime="text/csv",
            )
        else:
            st.success("🎉 No threats detected in this dataset!")

    with tab_safe:
        safes = df_res[df_res["Prediction"] == "Safe"][SHOW_COLS].reset_index(drop=True)
        st.dataframe(safes, use_container_width=True, height=380,
                     column_config={
                         "Threat_Prob_%": st.column_config.ProgressColumn(
                             "Threat %", min_value=0, max_value=100, format="%.1f%%"),
                         "Confidence_%": st.column_config.ProgressColumn(
                             "Confidence", min_value=0, max_value=100, format="%.1f%%"),
                     })
        st.download_button(
            "⬇️  Download Safe Records Only",
            data=safes.to_csv(index=False).encode("utf-8"),
            file_name="safe_records.csv", mime="text/csv",
        )

    with tab_model:
        t = st.session_state.trained
        ma1, ma2, ma3 = st.columns(3)
        ma1.metric("Validation Accuracy", f"{t['acc']*100:.2f}%")
        ma2.metric("ROC-AUC",             f"{t['roc_auc']:.4f}")
        ma3.metric("Features Used",       len(t["feature_cols"]))

        mc1, mc2 = st.columns(2)
        with mc1:
            st.plotly_chart(confusion_heatmap(t["cm"]),
                            use_container_width=True, config={"displayModeBar": False})
        with mc2:
            st.plotly_chart(roc_chart(t["fpr"], t["tpr"], t["roc_auc"]),
                            use_container_width=True, config={"displayModeBar": False})

        top_n = st.slider("Show top N features", 5, min(30, len(t["feature_cols"])), 15)
        st.plotly_chart(
            importance_chart(t["importances"], t["feature_cols"], top_n),
            use_container_width=True, config={"displayModeBar": False},
        )


# ═══════════════════════════════════════════════════════════════
# EMPTY STATE  (no predictions yet)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.trained is None:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:48px 0;color:#8b949e;">
      <div style="font-size:3rem;margin-bottom:12px">📂</div>
      <div style="font-family:'JetBrains Mono';font-size:1rem;color:#e6edf3;margin-bottom:6px">
        Get started in 2 steps
      </div>
      <div style="font-size:.9rem;line-height:1.8">
        1 · Upload your <b style="color:#58a6ff">train.csv</b> and click <b>Train Model</b><br>
        2 · Upload any CSV and click <b>Run Predictions</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
