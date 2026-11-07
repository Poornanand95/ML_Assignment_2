"""
ShopSense - purchase-intent classifier dashboard (Streamlit).

Loads the five pre-trained pipelines from model/ and evaluates them on a
user-uploaded test CSV. No model is re-trained inside the app.

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from shopper_utils import (
    FEATURE_COLS,
    MODEL_FILES,
    TARGET_COL,
    clean_frame,
    evaluate,
    missing_columns,
    positive_scores,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"
CLASS_NAMES = ["No purchase (0)", "Purchase (1)"]
ACCENT = "#2a78d6"
MUTED = "#b4b2a9"

st.set_page_config(page_title="ShopSense | Purchase Intent", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {border-left: 6px solid #2a78d6; padding: .6rem 1rem; margin-bottom: 1rem;
             background: rgba(42,120,214,.07); border-radius: 6px;}
      .hero h1 {margin: 0; font-size: 1.9rem;}
      .hero p {margin: .2rem 0 0; opacity: .8;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
             border-radius: 8px; padding: .5rem .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading saved models ...")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_training_report() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def read_csv(data: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(data))


def score_model(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    metrics = evaluate(y, y_pred, y_score) if y.nunique() == 2 else None
    return {"y_pred": y_pred, "y_score": y_score, "metrics": metrics}


# ---------------------------------------------------------------- charts
def confusion_chart(cm: np.ndarray) -> alt.Chart:
    rows = [
        {"Actual": CLASS_NAMES[i], "Predicted": CLASS_NAMES[j], "Count": int(cm[i, j]),
         "Share": cm[i, j] / cm[i].sum() if cm[i].sum() else 0}
        for i in range(2) for j in range(2)
    ]
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=CLASS_NAMES, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=CLASS_NAMES),
    )
    cells = base.mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual", "Predicted", "Count", alt.Tooltip("Share:Q", format=".1%", title="Row %")],
    )
    labels = base.mark_text(fontSize=18, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(alt.datum.Share > 0.5, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (cells + labels).properties(height=300)


def roc_chart(y_true, y_score, auc_value: float) -> alt.Chart:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    curve = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
    line = alt.Chart(curve).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("False positive rate:Q", format=".3f"),
                 alt.Tooltip("True positive rate:Q", format=".3f")],
    )
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color=MUTED, strokeDash=[4, 4], strokeWidth=1
    ).encode(x="x:Q", y="y:Q")
    return (chance + line).properties(height=300, title=f"ROC curve (AUC = {auc_value:.3f})")


def comparison_chart(table: pd.DataFrame, metric: str, highlight: str) -> alt.Chart:
    data = table[[metric]].reset_index().rename(columns={"index": "Model"})
    data["Selected"] = np.where(data["Model"] == highlight, "Selected model", "Other models")
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[min(0, data[metric].min()), 1])),
        y=alt.Y("Model:N", sort="-x", title=None),
        color=alt.Color("Selected:N", scale=alt.Scale(domain=["Selected model", "Other models"],
                                                      range=[ACCENT, MUTED]), legend=None),
        tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
    )
    text = bars.mark_text(align="left", dx=4).encode(text=alt.Text(f"{metric}:Q", format=".3f"),
                                                     color=alt.value("#52514e"))
    return (bars + text).properties(height=230)


# ---------------------------------------------------------------- sidebar
models = load_models()
train_report = load_training_report()

with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES), index=4)
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"],
                                help="Must contain the 17 feature columns and the 'Revenue' target.")
    st.caption("No file handy? Download the held-out test split and upload it back:")
    if SAMPLE_TEST.exists():
        st.download_button("⬇️ Download test_data.csv", SAMPLE_TEST.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", use_container_width=True)
    use_sample = st.toggle("Use bundled test_data.csv instead", value=False,
                           disabled=uploaded is not None)
    st.divider()
    st.caption("Models were trained offline by `train_models.py`. "
               "The app only loads them and predicts - nothing is re-trained here.")

st.markdown(
    """
    <div class="hero">
      <h1>🛒 ShopSense: Will this visitor buy?</h1>
      <p>Comparing five classifiers on the UCI <i>Online Shoppers Purchasing Intention</i> dataset
      &middot; ML Assignment 2 (M.Tech AIML, BITS WILP)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- data
if uploaded is not None:
    raw_df, source = read_csv(uploaded.getvalue()), uploaded.name
elif use_sample and SAMPLE_TEST.exists():
    raw_df, source = pd.read_csv(SAMPLE_TEST), "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV in the sidebar (or switch on the bundled test data) to evaluate the models.")
    with st.expander("Expected CSV columns"):
        st.code(", ".join(FEATURE_COLS + [TARGET_COL]))
    if train_report:
        st.subheader("Reference results on the held-out test split")
        ref = pd.DataFrame({m: r["metrics"] for m, r in train_report["models"].items()}).T
        st.dataframe(ref.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)
    st.stop()

df = clean_frame(raw_df)
missing = missing_columns(df, need_target=True)
if missing:
    st.error(f"The uploaded file is missing required column(s): **{', '.join(missing)}**")
    st.stop()

before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
if len(df) < before:
    st.warning(f"Dropped {before - len(df)} row(s) with missing or non-numeric values.")

X, y = df[FEATURE_COLS], df[TARGET_COL]

c1, c2, c3 = st.columns(3)
c1.metric("Source", source)
c2.metric("Rows evaluated", f"{len(df):,}")
c3.metric("Purchase rate", f"{y.mean():.1%}")

if y.nunique() < 2:
    st.warning("The target column contains only one class, so AUC and MCC cannot be computed. "
               "Upload a file with both purchase and non-purchase sessions.")

result = score_model(models[model_name], X, y)

tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔎 Predictions", "ℹ️ About"]
)

# ---------------------------------------------------------------- tab 1
with tab_eval:
    st.subheader(f"{model_name}")
    if result["metrics"]:
        cols = st.columns(6)
        for col, (k, v) in zip(cols, result["metrics"].items()):
            col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, result["y_pred"], labels=[0, 1])
        st.altair_chart(confusion_chart(cm), use_container_width=True)
    with right:
        if result["metrics"]:
            st.altair_chart(roc_chart(y, result["y_score"], result["metrics"]["AUC"]),
                            use_container_width=True)

    st.markdown("**Classification report**")
    rep = classification_report(y, result["y_pred"], labels=[0, 1], target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    rep_df["support"] = rep_df["support"].astype(int)
    st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                      "f1-score": "{:.4f}", "support": "{:,}"}),
                 use_container_width=True)

# ---------------------------------------------------------------- tab 2
with tab_compare:
    if y.nunique() < 2:
        st.info("Comparison needs both classes in the target column.")
    else:
        all_scores = {name: score_model(m, X, y)["metrics"] for name, m in models.items()}
        table = pd.DataFrame(all_scores).T
        st.markdown("**All five models on the uploaded data** (darker = better within each column)")
        st.dataframe(table.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)
        metric = st.radio("Metric to plot", list(table.columns), index=5, horizontal=True)
        st.altair_chart(comparison_chart(table, metric, model_name), use_container_width=True)
        best = table["MCC"].idxmax()
        st.success(f"Best by MCC on this file: **{best}** (MCC = {table.loc[best, 'MCC']:.4f}, "
                   f"F1 = {table.loc[best, 'F1']:.4f})")

# ---------------------------------------------------------------- tab 3
with tab_pred:
    preview = raw_df.loc[df.index].copy()
    preview.insert(0, "Predicted", result["y_pred"])
    preview.insert(1, "Score", np.round(result["y_score"], 4))
    preview.insert(2, "Correct", result["y_pred"] == y.values)
    only_wrong = st.checkbox("Show misclassified rows only")
    shown = preview[~preview["Correct"]] if only_wrong else preview
    st.caption(f"{len(shown):,} rows")
    st.dataframe(shown, use_container_width=True, height=400)
    st.download_button("⬇️ Download predictions", preview.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- tab 4
with tab_about:
    st.markdown(
        """
        **Task.** Predict whether an e-commerce browsing session ends in a purchase
        (`Revenue` = 1) from 17 session features: page counts and durations
        (administrative, informational, product-related), Google Analytics bounce/exit
        rates and page value, closeness to a special day, month, OS, browser, region,
        traffic type, visitor type and weekend flag.

        **Preprocessing (inside every saved pipeline).** `log1p` + standard scaling for
        the 10 numeric columns, one-hot encoding for the 6 categorical columns, weekend
        flag passed through. Because only ~15 % of sessions convert, hyper-parameters were
        tuned with 5-fold stratified CV on **F1**, and Logistic Regression, Decision Tree
        and SVM use balanced class weights.
        """
    )
    if train_report:
        params = pd.DataFrame(
            {m: {"Best hyper-parameters": ", ".join(f"{k}={v}" for k, v in r["best_params"].items()),
                 "CV F1": r["cv_f1"]} for m, r in train_report["models"].items()}
        ).T
        st.dataframe(params, use_container_width=True)
        st.caption(f"Train rows: {train_report['n_train']:,} · Test rows: {train_report['n_test']:,} "
                   f"· Random seed: {train_report['seed']}")
