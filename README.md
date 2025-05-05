# 💳 Fraud Detection in Financial Transactions

**Author:** Deepak250104

**Dataset:** [Google Drive Link](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)

**Model:** LightGBM (Gradient Boosted Trees)

**Language:** Python 3.12.4

**Frameworks:** scikit-learn, LightGBM, pandas, matplotlib, seaborn

---

## 📘 Project Overview

This project focuses on detecting fraudulent financial transactions using a machine learning approach. The dataset contains over 6.3 million records simulating anonymized transactions, with 10 engineered and raw features. The goal is to accurately classify transactions as fraudulent or legitimate using interpretable and scalable methods. The system is designed with modular notebooks and can be deployed in production environments.

---

## 📁 Project Structure

```
Fraud_Detection_in_Financial_Transactions/
├── data/
│   ├── Fraud.csv                  # Original dataset
│   ├── processed.csv              # Preprocessed data
│   ├── processed_features.csv     # Features after preprocessing
│   ├── processed_target.csv       # Target labels
│   └── Data Dictionary.txt        # Feature descriptions
│
├── models/
│   ├── lgbm_model.pkl             # Trained LightGBM model
│   └── preprocessor.pkl           # Data preprocessing pipeline
│
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── feature_extraction.ipynb  # Feature engineering and transformations
│   ├── model_selection.ipynb     # Model training and hyperparameter tuning
│   └── evaluation.ipynb          # Model performance and final validation
│
├── LICENSE
├── README.md
└── requirements.txt              # All dependencies
```

---

## 🔢 Dataset Summary

* **Size:** \~6.3 million rows
* **Features:** 10 columns including `amount`, `oldbalanceOrg`, `newbalanceOrig`, `type`, etc.
* **Target Variable:** `isFraud` (0 = normal, 1 = fraudulent)
* **Class Imbalance:**

  * **Fraudulent transactions:** \~0.129% of total data
  * This extreme imbalance was addressed using:

    * `class_weight="balanced"` in LightGBM
    * Stratified sampling during cross-validation
    * Emphasis on precision, recall, and AUC instead of accuracy

---

## 🧹 Data Cleaning & Preprocessing

* **Missing Values:** None in the dataset, verified in EDA.
* **Outliers:** Tree-based models handle them well; no removal needed.
* **Multicollinearity:** Not problematic for LightGBM; no explicit removal.
* **Dropped Columns:**

  * `nameOrig`, `nameDest` (identifiers)
  * `isFlaggedFraud` (data leakage)
* **Engineered Features:**

  * `errorBalanceOrig = oldbalanceOrg - newbalanceOrig - amount`
  * `errorBalanceDest = newbalanceDest - oldbalanceDest - amount`
    These features capture inconsistencies in the transaction flow and significantly improve predictive power.

---

## 🧠 Model Description

We used **LightGBM**, a gradient boosting algorithm optimized for performance and scale. Key reasons for selection:

* Handles large-scale data efficiently
* Robust to skewed distributions and multicollinearity
* Supports automatic handling of categorical features
* Fast training and inference

### 🔍 Hyperparameter Optimization

Used `RandomizedSearchCV` on parameters:

* `num_leaves`
* `learning_rate`
* `n_estimators`
* `max_depth`
* `min_child_samples`

Cross-validation ensured that tuning was not overfit to a single train-test split.

---

## 📊 Model Performance

Achieved **outstanding results** on the holdout set:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 100.0% |
| Precision | 100.0% |
| Recall    | 100.0% |
| F1-Score  | 1.000  |
| AUC-ROC   | 0.9998 |

* **Confusion Matrix**: No false positives or false negatives
* **ROC Curve**: AUC close to 1, indicating near-perfect separation between fraud and non-fraud

> ⚠️ Caution: These metrics may indicate possible data leakage or overfitting. All pipeline steps, especially preprocessing and feature engineering, were verified to avoid this.

---

## 🔍 Key Predictive Factors

* **Transaction Type**: `TRANSFER` and `CASH_OUT` were frequently associated with fraud
* **Amount**: Extremely high or low transaction values often flagged as suspicious
* **Balance Mismatch Features**: Engineered error features revealed fraudulent behavior when account balances didn’t reflect legitimate transaction logic

These factors align with common fraud patterns, such as draining balances, mimicking real user behavior, or inconsistent balance states.

---

## 🛡️ Infrastructure Recommendations

To prevent fraud effectively, financial institutions should:

* **Deploy real-time fraud detection models** like this one
* **Log model decisions** for auditability and compliance
* **Update models periodically** to adapt to evolving fraud tactics
* **Use human-in-the-loop systems** for high-stake decisions
* **Ensure secure feature engineering** to prevent leakage from future data or labels

---

## 📈 Post-deployment Monitoring

Once integrated, success should be measured by:

* Reduction in fraud losses
* Monitoring false positives (blocked genuine users)
* Real-world precision/recall vs validation scores
* Feedback from manual investigation teams
* A/B testing against baseline systems

---

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. `EDA.ipynb`
2. `feature_extraction.ipynb`
3. `model_selection.ipynb`
4. `evaluation.ipynb`

---

## 🔐 License

This project is licensed under the MIT License.

## Author 

[Deepak250104](https://github.com/Deepak250104)
