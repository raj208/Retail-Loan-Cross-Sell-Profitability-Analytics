# 🏦 Retail Loan Cross-Sell & Profitability Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

### **Executive Summary**
This project is an end-to-end Machine Learning solution designed for a Retail Banking context. Instead of simply predicting "who will buy a loan" (Propensity) or "who will default" (Risk) in isolation, this system combines both models to optimize for **Expected Profit**.

It features a dual-model architecture trained on the LendingClub dataset (2M+ records) and is deployed via an interactive Streamlit dashboard that allows relationship managers to simulate credit scenarios in real-time.

---

## 🏗️ Architecture & Logic

The system utilizes a **"Dual-Model Strategy"** to filter customers through two rigorous gates:

1.  **Stage 1: Eligibility Model (Propensity)**
    * **Goal:** Predict if a customer qualifies for a loan (Approval Probability).
    * **Algorithm:** XGBoost Classifier.
    * **Data:** Trained on a mix of Accepted Loans and Rejected Applications.
    * **Performance:** ~94% Accuracy.

2.  **Stage 2: Risk Model (Probability of Default - PD)**
    * **Goal:** Predict the likelihood of a borrower defaulting on the loan.
    * **Algorithm:** XGBoost Classifier (with class weights for imbalance).
    * **Data:** Trained on mature loans (Fully Paid vs. Charged Off).
    * **Performance:** ROC-AUC 0.71 (Industry Standard).

### **The Business Logic (Profitability Engine)**
We move beyond raw predictions to calculated **Net Expected Profit (NEP)** per customer:

$$
NEP = P(Eligible) \times [ (Interest Income \times (1 - PD)) - (LGD \times PD) ] - Acquisition Cost
$$

* **PD:** Probability of Default (from Stage 2 Model)
* **LGD:** Loss Given Default (Assumed 60%)
* **Acquisition Cost:** Fixed cost per lead.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Deployment:** Streamlit
* **Development Environment:** Google Colab (for heavy training) + VS Code (for app development)

---

## 📂 Project Structure

```text
loan_analytics_project/
│
├── app/
│   └── streamlit_app.py        # Main Dashboard Application
│
├── data/
│   ├── raw/                    # (Not uploaded to git) Raw CSVs
│   └── processed/              # Cleaned data for the dashboard
│
├── models/
│   ├── pd_model.pkl            # Trained Risk Model
│   ├── propensity_model.pkl    # Trained Eligibility Model
│   └── *_cols.pkl              # Column metadata for reproducibility
│
├── notebooks/
│   ├── 01_Data_Ingestion.ipynb
│   ├── 02_EDA_and_Insights.ipynb
│   └── ...
│
├── screenshots/                # Images for README
│   ├── dashboard_overview.png
│   └── simulator_demo.png
│
├── requirements.txt            # Dependencies
└── README.md                   # Project Documentation