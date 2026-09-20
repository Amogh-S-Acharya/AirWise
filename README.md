# 🌍 AirWise

## Explainable and Uncertainty-Aware Air Quality Forecasting and City Profiling

AirWise is a machine-learning based air-quality intelligence system that forecasts **next-day Air Quality Index (AQI)**, estimates prediction uncertainty, explains model predictions using **SHAP**, and profiles Indian cities using **K-Means clustering and PCA**.

---

## 🚀 Features

* 📈 Next-day AQI forecasting
* 📊 Quantile regression using LightGBM
* 🔮 P10 / P50 / P90 uncertainty estimation
* 🤖 Ensemble learning
* Random Forest
* LightGBM
* Gradient Boosting
* 🔍 SHAP explainability
* 🏙️ K-Means city clustering
* 📉 PCA visualization
* 🌦️ Seasonal AQI heatmap
* 🖥️ Interactive Streamlit dashboard

---

## 🧠 Methodology

### 1. AQI Forecasting

Historical AQI and pollutant observations are transformed into temporal features:

* AQI lag 1
* AQI lag 7
* 7-day rolling AQI
* PM2.5 lag 1
* PM10 lag 1
* NO2 lag 1
* CO lag 1
* Month
* Day of week

The target is the **next day's AQI**.

### 2. Uncertainty Estimation

Three LightGBM quantile models are trained:

* **P10** — lower prediction
* **P50** — median prediction
* **P90** — upper prediction

The P10–P90 range represents the model's prediction interval.

### 3. Ensemble Learning

The following models are evaluated:

* Random Forest
* LightGBM
* Gradient Boosting

Their predictions are also combined using an averaging ensemble.

### 4. Explainable AI

**SHAP (SHapley Additive exPlanations)** is used to understand the contribution of individual features to model predictions.

### 5. City Profiling

Cities are represented using:

* Mean AQI
* Mean PM2.5
* Mean PM10
* Winter AQI
* Monsoon AQI

The features are standardized before applying **K-Means clustering**. **PCA** is then used to visualize the resulting city groups in two dimensions.

---

## 📊 Dataset

**Air Quality Data in India (2015–2020)**

Dataset source:

[Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The downloaded dataset contains:

* **29,531 records**
* **26 cities**
* **16 columns**
* Date range: **2015-01-01 to 2020-07-01**

Important variables include:

`PM2.5`, `PM10`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, and `AQI_Bucket`.

---

## 🧪 Experimental Setup

The forecasting experiment uses an **80/20 chronological train-test split**.

### Delhi Experiment

**Training period**

```text
2015-01-08 → 2019-06-11
```

**Testing period**

```text
2019-06-12 → 2020-06-30
```

Chronological splitting is used to avoid using future observations for training when evaluating earlier observations.

---

## 📈 Results

### Delhi Quantile LightGBM

| Metric                 |  Value |
| ---------------------- | -----: |
| MAE                    |  43.19 |
| RMSE                   |  59.47 |
| R²                     |  0.735 |
| P10–P90 Coverage       | 55.84% |
| Average Interval Width |  85.82 |

The P10–P90 interval achieved **55.84% empirical coverage** on the Delhi test set, indicating that the raw uncertainty interval requires further calibration.

### Model Comparison

| Model             |   MAE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: |
| Random Forest     | 44.41 | 61.07 | 0.720 |
| LightGBM          | 43.85 | 59.89 | 0.731 |
| Gradient Boosting | 44.68 | 62.24 | 0.709 |
| Ensemble Average  | 43.65 | 60.11 | 0.729 |

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard providing:

1. City selection
2. AQI forecasting
3. Prediction uncertainty
4. SHAP explanations
5. City clustering
6. PCA visualization
7. Seasonal AQI analysis
8. Model comparison

### Run the application

This project uses `uv`.

```bash
uv sync
```

Then:

```bash
uv run streamlit run app.py
```

No manual virtual-environment activation is required when using `uv run`.

---

## 📁 Project Structure

```text
AirWise/
│
├── data/
│   ├── raw/
│   │   └── city_day.csv
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── data_prep.py
│   ├── forecast.py
│   ├── ensemble.py
│   ├── clustering.py
│   ├── seasonal.py
│   └── explain.py
│
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AirWise.git
cd AirWise
```

Install dependencies:

```bash
uv sync
```

Run the Streamlit application:

```bash
uv run streamlit run app.py
```

---

## ▶️ Run Individual Components

### Data Preparation

```bash
uv run python src/data_prep.py
```

### AQI Forecasting

```bash
uv run python src/forecast.py
```

### Explainability

```bash
uv run python src/explain.py
```

### Ensemble Learning

```bash
uv run python src/ensemble.py
```

### City Clustering

```bash
uv run python src/clustering.py
```

### Seasonal Analysis

```bash
uv run python src/seasonal.py
```

---

## ⚠️ Limitations

* Dataset ends in 2020.
* Meteorological variables are not included.
* Raw quantile intervals are not fully calibrated.
* Current forecasting models are city-specific.
* Missing observations require preprocessing.
* The ensemble currently uses simple averaging.
* AQI is an aggregate indicator and does not represent a complete atmospheric or health-risk model.

---

## 🔮 Future Scope

* Conformal prediction for better uncertainty calibration
* Integration of meteorological variables
* Real-time AQI data
* LSTM / GRU / Transformer models
* Multi-city forecasting
* Advanced stacking and weighted ensembles
* Real-time AQI early-warning system
* Improved uncertainty modeling

---

## 👨‍💻 Project

**AirWise** is an academic machine-learning project demonstrating:

* Machine Learning
* Ensemble Learning
* Quantile Regression
* Uncertainty Estimation
* Explainable AI
* SHAP
* K-Means Clustering
* PCA
* Time-Series Feature Engineering
* Interactive ML Deployment with Streamlit

---

## 📌 Note

The primary experimental results reported in this repository are based on the **Delhi chronological test experiment**. The Streamlit application additionally demonstrates multi-city functionality.
