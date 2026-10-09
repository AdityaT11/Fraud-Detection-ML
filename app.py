
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud Detection ML Platform",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models"

MODELS = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
}

st.title("🛡️ Fraud Detection ML Platform")
st.caption("Transaction risk assessment and model evaluation")

st.warning(
    "Educational project only. Predictions should not be used alone "
    "to block real financial transactions."
)

# Load actual evaluation results
metrics_path = REPORTS / "model_comparison.csv"

if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
else:
    metrics_df = pd.DataFrame()

prediction_tab, evaluation_tab, threshold_tab = st.tabs(
    ["Transaction Assessment", "Model Evaluation", "Threshold Analysis"]
)

with prediction_tab:
    st.subheader("Transaction details")

    with st.sidebar:
        st.header("Model configuration")
        model_name = st.selectbox("Model", list(MODELS))
        threshold = st.slider(
            "Fraud classification threshold",
            min_value=0.05,
            max_value=0.99,
            value=0.90,
            step=0.01,
        )

    left, right = st.columns(2)

    with left:
        step = st.number_input("Time step", min_value=1, value=1)
        transaction_type = st.selectbox(
            "Transaction type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        )
        amount = st.number_input("Transaction amount", min_value=0.0, value=1000.0)
        old_org = st.number_input("Sender balance before", min_value=0.0, value=5000.0)

    with right:
        new_org = st.number_input("Sender balance after", min_value=0.0, value=4000.0)
        old_dest = st.number_input("Receiver balance before", min_value=0.0, value=0.0)
        new_dest = st.number_input("Receiver balance after", min_value=0.0, value=1000.0)

    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }])

    if st.button("Assess transaction", type="primary"):
        model_path = MODELS[model_name]

        if not model_path.exists():
            st.error(f"Model not found: {model_path}")
        else:
            try:
                model = joblib.load(model_path)
                score = float(model.predict_proba(transaction)[0, 1])

                a, b = st.columns(2)
                a.metric("Estimated fraud score", f"{score:.2%}")
                b.metric("Decision threshold", f"{threshold:.0%}")

                if score >= threshold:
                    st.error("Flagged for fraud review")
                else:
                    st.success("Not flagged at this threshold")

                st.progress(score)
                st.caption(
                    "Model scores may not be calibrated probabilities. "
                    "The selected threshold is configurable and requires "
                    "validation for the intended use."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with evaluation_tab:
    st.subheader("Model performance comparison")

    if metrics_df.empty:
        st.info("Run train.py first to generate evaluation metrics.")
    else:
        preferred = [
            "Random Forest",
            "Logistic Regression",
        ]
        choices = [m for m in preferred if m in metrics_df["model"].values]
        selected = st.selectbox("View model metrics", choices)

        row = metrics_df.loc[metrics_df["model"] == selected].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{row['precision']:.2%}")
        c2.metric("Recall", f"{row['recall']:.2%}")
        c3.metric("F1-score", f"{row['f1_score']:.3f}")

        c4, c5 = st.columns(2)
        c4.metric("PR-AUC", f"{row['pr_auc']:.4f}")
        c5.metric("ROC-AUC", f"{row['roc_auc']:.4f}")

        st.markdown("#### Metric comparison")
        metric_options = ["precision", "recall", "f1_score", "pr_auc", "roc_auc"]
        metric_choice = st.selectbox("Metric to compare", metric_options)

        chart_df = metrics_df.set_index("model")[[metric_choice]]
        st.bar_chart(chart_df)

        image_path = REPORTS / (
            selected.lower().replace(" ", "_") + "_confusion.png"
        )

        st.markdown("#### Confusion matrix")
        if image_path.exists():
            st.image(str(image_path), caption=f"{selected} — test-set confusion matrix")
        else:
            st.info("Confusion matrix image is not available.")

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.caption(
            "These metrics come from the saved model-comparison report. "
            "The fraud class is rare, so accuracy alone would be misleading."
        )

with threshold_tab:
    st.subheader("Threshold analysis on validation data")

    threshold_path = REPORTS / "threshold_analysis.csv"

    if threshold_path.exists():
        threshold_df = pd.read_csv(threshold_path)
        st.line_chart(
            threshold_df.set_index("threshold")[["precision", "recall", "f1"]]
        )
        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Validation results only. Choose a threshold according to the "
            "cost of false alarms versus missed fraud; the best F1 threshold "
            "is not necessarily best for every application."
        )
    else:
        st.info("Run tune_threshold.py to generate threshold-analysis results.")

st.divider()
st.caption("Fraud Detection ML Platform • Portfolio demonstration")

