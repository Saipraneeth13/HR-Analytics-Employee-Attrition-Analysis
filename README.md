# 📊 HR Employee Attrition Prediction & Retention Analytics

An end-to-end machine learning and people-analytics project built on the IBM HR Analytics dataset (`1,470` employee records, `35` features) to uncover key drivers of employee turnover and predict flight risk prior to resignation.

---

## 🔍 Key Analytical Findings

* **Overall Attrition Rate:** **16.12%** (`237` departures vs. `1,233` retained employees).
* **Overtime Impact:** Employees working overtime resign at **30.53%**, nearly **3×** the rate of non-overtime staff (**10.44%**).
* **Compensation Gap:** Departing employees earn a median monthly income of **$3,202** compared to **$5,204** for retained staff.
* **High-Risk Roles:** **Sales Representatives** experience the highest turnover (**39.76%**), followed by **Laboratory Technicians** (**23.94%**) and **Human Resources** (**23.08%**).
* **Early-Tenure Vulnerability:** Median company tenure for departing employees is **3 years** (vs. **6 years** for retained employees).

![Attrition by Role and Overtime](reports/figures/eda_role_overtime.png)

---

## 📈 Model Benchmark Results

Because missing a departing employee is significantly costlier than triggering a proactive retention check-in, models were evaluated using **ROC-AUC**, **PR-AUC**, and **Recall** on a stratified 80/20 test split (`294` employees):

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | PR-AUC | Recall (Attrition) | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.8270** | **0.8076** | **0.5690** | **0.6809** | 0.3810 | **0.4885** |
| Gradient Boosting | 0.8040 | 0.7972 | 0.4600 | 0.1915 | **0.5294** | 0.2812 |
| Random Forest (Balanced) | 0.7960 | 0.7778 | 0.4294 | 0.1064 | 0.4545 | 0.1724 |

![ROC Curves and Confusion Matrix](reports/figures/model_evaluation.png)

---

## 🗂️ Repository Structure

```text
hr-employee-attrition-prediction/
├── app/
│   └── streamlit_app.py              # Interactive HR risk simulator web app
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── processed/
│       └── hr_attrition_processed.csv
├── models/
│   └── attrition_model.joblib        # Serialized scikit-learn pipeline
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb     # Exploratory analysis & model notebook
├── reports/
│   ├── figures/                      # High-res EDA & evaluation plots
│   └── model_metrics.json            # Exported benchmark metrics
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data cleaning & domain feature engineering
│   ├── train.py                      # Cross-validation & model training script
│   └── predict.py                    # CLI batch risk-scoring script
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/hr-employee-attrition-prediction.git
cd hr-employee-attrition-prediction
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
```bash
python -m src.train
```

### 3. Score Employees via CLI
```bash
python -m src.predict --input data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --output reports/employee_risk_scores.csv
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 💡 Strategic HR Recommendations
1. **Audit Overtime in High-Attrition Roles:** Rebalance workloads or introduce compensatory time-off policies for `Sales Representatives` and `Laboratory Technicians` working frequent overtime.
2. **Entry-Level Compensation Review:** Benchmark pay bands for employees earning under **$3,500/month**, where attrition risk spikes sharply.
3. **Year-2 Career Checkpoints:** Implement structured mentorship and promotion readiness reviews between Years 2 and 3 to curb early-tenure exits.
