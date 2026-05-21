# 🎯 Marketing Campaign Performance Prediction

## 📌 Project Overview

An end-to-end machine learning project analyzing and predicting marketing campaign performance for three major beauty brands — Nykaa, Purplle, and Tira.

## 🎯 Problem Statement

Marketing teams generate large volumes of campaign data including impressions, clicks, conversions, acquisition costs, and revenue. This project transforms raw campaign data into actionable insights and builds ML models to predict campaign Revenue, ROI, and Profit/Loss.

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | MySQL |
| Visualizations | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |

## 📁 Project Structure
marketing_campaign_project/
│
├── data/
│ ├── raw/ # Original CSV files
│ └── processed/ # Cleaned & featured data
│
├── notebooks/
│ ├── 01_data_loading.ipynb # Data cleaning & preprocessing
│ ├── 02_database.ipynb # MySQL connection & storage
│ ├── 03_eda.ipynb # Exploratory Data Analysis
│ ├── 04_feature_engineering.ipynp # Encoding & feature selection
│ └── 05_modeling.ipynb # Model building & evaluation
│
├── models/ # Saved model files
├── app/
│ └── app.py # Streamlit application
└── README.md


## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| Original rows per brand | 55,555 |
| Combined rows after cleaning | 52,439 |
| Number of brands | 3 |
| Features after engineering | 24 |

## 🔄 ML Pipeline Steps

1. **Data Collection** — Load 3 CSVs, merge into one DataFrame
2. **Data Preprocessing** — Handle nulls, fix dtypes, remove duplicates
3. **Feature Engineering** — Recalculate ROI, create Profit_Flag, MultiLabelBinarizer for Channel_Used
4. **EDA** — Brand, Campaign Type, Channel wise analysis
5. **Model Building** — Regression & Classification with sklearn Pipelines
6. **Deployment** — Streamlit app for interactive predictions

## 🤖 Model Performance

### Regression (Revenue Prediction)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 2,80,989 | 1,78,799 | 0.6677 |
| Decision Tree | 3,89,428 | 2,40,300 | 0.3618 |
| Random Forest | 2,71,402 | 1,72,780 | **0.6900** ✅ |

### Regression (ROI Prediction)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 10,44,854 | 4,56,165 | 0.4643 |
| Decision Tree | 8,51,353 | 2,97,630 | 0.6444 |
| Random Forest | 6,04,378 | 2,15,244 | **0.8208** ✅ |

### Classification (Profit/Loss Prediction)

| Model | Accuracy | F1-Score (Profit) |
|-------|----------|-------------------|
| Logistic Regression | 0.52 | 0.68 |
| Decision Tree | 0.90 | **0.95** ✅ |
| Random Forest | 0.95 | 0.97 |

## 🚀 How to Run

### Step 1: Clone Repository
```bash
git clone https://github.com/ShaliniS0309/marketing-campaign-performance-prediction.git cd marketing-campaign-performance
```
### Step 1: Clone Repository
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux 
### Step 2:Create Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
### Step 3: Install Dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn sqlalchemy pymysql streamlit jupyter ipykernel imbalanced-learn
### Step 4:Run Streamlit App
streamlit run app/app.py


## 📈 Key Insights

Nykaa leads across Revenue, Conversions, and ROI

Email campaigns generate highest revenue and conversions

Marketing funnel works correctly — Impressions → Clicks → Leads → Conversions → Revenue all positively correlated

Random Forest is the best model for both Revenue (R²=0.69) and ROI (R²=0.82) prediction.


## ⚠️ Known Limitations

Issue	Description
Class Imbalance	Classification model struggles with Loss detection
Loss Definition	Loss campaigns derived from null Revenue values filled with 0
High ROI Values	ROI values are very high due to scale difference between Revenue and Acquisition Cost.



