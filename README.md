# Fraud Detection ML Platform

A machine learning application for detecting potentially fraudulent financial transactions using an interactive Streamlit dashboard, model comparison, and decision-threshold analysis.

## Overview

Financial fraud detection is challenging because fraudulent transactions represent a very small proportion of total transactions. This project explores supervised machine learning techniques to identify suspicious transactions while evaluating the trade-off between detecting fraud and generating false alarms.

The platform allows users to enter transaction details, select a trained model, adjust the classification threshold, and review model evaluation metrics.

## Features

- Interactive transaction assessment using Streamlit
- Logistic Regression and Random Forest model comparison
- Numerical feature scaling and categorical feature encoding
- Training-only random undersampling for class imbalance
- Evaluation using Precision, Recall, F1-score, PR-AUC, and ROC-AUC
- Confusion matrix visualizations
- Validation-based threshold analysis
- Automated preprocessing tests

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn, imbalanced-learn
- **Visualization:** Matplotlib, Streamlit
- **Model persistence:** Joblib
- **Testing:** Pytest

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

Expected input features include:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

The target column is `isFraud`.

Download the dataset from [Kaggle — PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the full CSV at `data/transactions.csv`.

The dataset is not included in this repository because of its size. Review the dataset's terms and limitations before reuse.

## Project Structure

```text
Fraud-Detection-ML/
├── app.py
├── train.py
├── tune_threshold.py
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── data/
├── models/
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Fraud-Detection-ML
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Models

After placing the dataset at `data/transactions.csv`, run:

```bash
python train.py --data data/transactions.csv
```

The training pipeline compares Logistic Regression and Random Forest. It saves trained models in `models/` and evaluation reports in `reports/`.

## Run Threshold Analysis

```bash
python tune_threshold.py
```

This evaluates candidate classification thresholds on a validation split and saves the results to `reports/threshold_analysis.csv`.

## Launch the Dashboard

```bash
streamlit run app.py
```

Open the local URL printed in your terminal, usually `http://localhost:8501`.

The dashboard includes:

1. **Transaction Assessment:** estimates a fraud score for user-entered transaction details.
2. **Model Evaluation:** compares saved evaluation metrics and confusion matrices.
3. **Threshold Analysis:** visualizes the validation-set precision, recall, and F1-score trade-off.

## Experimental Results

The initial held-out test evaluation produced the following results at the default classification threshold of 0.50:

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 2.38% | 95.44% | 0.0464 | 0.5785 | 0.9898 |
| Random Forest | 9.19% | 99.70% | 0.1682 | 0.8857 | 0.9995 |

In the separate validation threshold experiment, the Random Forest achieved a precision of approximately 40.8%, recall of 92.9%, and F1-score of 0.568 at a threshold of 0.90.

These results come from different evaluation runs and should not be compared as if they were measured on the same split. They are experimental results, not guarantees of real-world performance.

## Limitations and Future Work

- Evaluate possible target leakage from transaction-balance features.
- Tune decision thresholds according to the costs of false positives and false negatives.
- Investigate probability calibration.
- Evaluate performance on a more realistic, independently sourced dataset.
- Add model monitoring and drift detection.
- Expand automated tests for training and prediction.

## Responsible Use

This is an educational portfolio project using synthetic data. Predictions should not be used as the sole basis for blocking real financial transactions. Real-world deployment requires further validation, privacy and security controls, monitoring, and human oversight.

## License

A license has not yet been selected. Add a license after deciding how you want others to use and redistribute this project.
