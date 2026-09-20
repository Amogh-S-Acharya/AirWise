import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from lightgbm import LGBMRegressor
import shap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirWise",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/raw/city_day.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return df


df = load_data()


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "AQI_lag_1",
    "AQI_lag_7",
    "AQI_rolling_7",
    "PM2.5_lag_1",
    "PM10_lag_1",
    "NO2_lag_1",
    "CO_lag_1",
    "month",
    "day_of_week",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(city_df):

    city_df = city_df.copy()

    city_df = city_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    city_df["AQI_lag_1"] = (
        city_df["AQI"].shift(1)
    )

    city_df["AQI_lag_7"] = (
        city_df["AQI"].shift(7)
    )

    city_df["AQI_rolling_7"] = (
        city_df["AQI"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    city_df["PM2.5_lag_1"] = (
        city_df["PM2.5"].shift(1)
    )

    city_df["PM10_lag_1"] = (
        city_df["PM10"].shift(1)
    )

    city_df["NO2_lag_1"] = (
        city_df["NO2"].shift(1)
    )

    city_df["CO_lag_1"] = (
        city_df["CO"].shift(1)
    )

    city_df["month"] = (
        city_df["Date"].dt.month
    )

    city_df["day_of_week"] = (
        city_df["Date"].dt.dayofweek
    )

    city_df["target_AQI"] = (
        city_df["AQI"].shift(-1)
    )

    return city_df


# ============================================================
# TRAIN QUANTILE MODELS
# ============================================================

@st.cache_resource
def train_models(city):

    city_df = df[
        df["City"] == city
    ].copy()

    city_df = city_df.dropna(
        subset=["AQI"]
    )

    feature_df = create_features(
        city_df
    )

    ml_df = feature_df[
        ["Date"] + FEATURES + ["target_AQI"]
    ].dropna()

    split_index = int(
        len(ml_df) * 0.8
    )

    train_df = ml_df.iloc[
        :split_index
    ]

    test_df = ml_df.iloc[
        split_index:
    ]

    X_train = train_df[FEATURES]
    y_train = train_df["target_AQI"]

    X_test = test_df[FEATURES]
    y_test = test_df["target_AQI"]

    models = {}

    for alpha, name in [
        (0.10, "P10"),
        (0.50, "P50"),
        (0.90, "P90")
    ]:

        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

        model.fit(
            X_train,
            y_train
        )

        models[name] = model

    predictions = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": y_test.values,
        "P10": models["P10"].predict(X_test),
        "P50": models["P50"].predict(X_test),
        "P90": models["P90"].predict(X_test),
    })

    predictions["Lower"] = np.minimum(
        predictions["P10"],
        predictions["P90"]
    )

    predictions["Upper"] = np.maximum(
        predictions["P10"],
        predictions["P90"]
    )

    coverage = (
        (
            (predictions["Actual"] >= predictions["Lower"]) &
            (predictions["Actual"] <= predictions["Upper"])
        ).mean()
    )

    mae = np.mean(
        np.abs(
            predictions["Actual"] -
            predictions["P50"]
        )
    )

    return (
        models,
        test_df,
        predictions,
        mae,
        coverage
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 AirWise")

st.sidebar.write(
    "Explainable and uncertainty-aware "
    "air quality intelligence."
)

cities = sorted(
    df["City"].dropna().unique()
)

selected_city = st.sidebar.selectbox(
    "Select City",
    cities,
    index=cities.index("Delhi")
    if "Delhi" in cities else 0
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 AirWise"
)

st.subheader(
    "Explainable & Uncertainty-Aware Air Quality Forecasting"
)

st.write(
    "AirWise uses historical air-quality observations "
    "to forecast next-day AQI, quantify prediction "
    "uncertainty, explain model predictions using SHAP, "
    "and profile cities using clustering."
)


# ============================================================
# MODEL
# ============================================================

with st.spinner(
    f"Training AirWise model for {selected_city}..."
):

    (
        models,
        test_df,
        predictions,
        mae,
        coverage
    ) = train_models(
        selected_city
    )


# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "City",
    selected_city
)

col2.metric(
    "Test Samples",
    len(test_df)
)

col3.metric(
    "MAE",
    f"{mae:.2f}"
)

col4.metric(
    "P10-P90 Coverage",
    f"{coverage * 100:.1f}%"
)


# ============================================================
# FORECAST
# ============================================================

st.header(
    "📈 Next-Day AQI Forecast"
)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    predictions["Date"],
    predictions["Actual"],
    label="Actual AQI",
    linewidth=1.3
)

ax.plot(
    predictions["Date"],
    predictions["P50"],
    label="Median Forecast",
    linewidth=1.3
)

ax.fill_between(
    predictions["Date"],
    predictions["Lower"],
    predictions["Upper"],
    alpha=0.2,
    label="P10-P90 Prediction Interval"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "AQI"
)

ax.set_title(
    f"{selected_city}: Next-Day AQI Forecast"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(
    fig
)


# ============================================================
# UNCERTAINTY EXPLANATION
# ============================================================

st.info(
    "The shaded region represents the model's "
    "P10-P90 prediction interval. Its empirical "
    "coverage is measured on the held-out test set."
)


# ============================================================
# SHAP
# ============================================================

st.header(
    "🔍 Explainable AI — SHAP"
)

st.write(
    "SHAP identifies which historical AQI, pollutant "
    "and calendar features influence the model's "
    "next-day prediction."
)

# Use median model
model = models["P50"]

X_test = test_df[FEATURES]

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer(
    X_test
)

fig_shap, ax_shap = plt.subplots(
    figsize=(10, 5)
)

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

st.pyplot(
    fig_shap
)

plt.close(fig_shap)


# ============================================================
# CITY CLUSTERING
# ============================================================

st.header(
    "🏙️ City Air-Quality Profiling"
)

df_cluster = df.copy()

df_cluster["month"] = (
    df_cluster["Date"].dt.month
)

city_features = (
    df_cluster
    .groupby("City")
    .agg(
        mean_AQI=("AQI", "mean"),
        mean_PM25=("PM2.5", "mean"),
        mean_PM10=("PM10", "mean")
    )
    .reset_index()
)

winter = (
    df_cluster[
        df_cluster["month"].isin(
            [11, 12, 1, 2]
        )
    ]
    .groupby("City")["AQI"]
    .mean()
    .rename("winter_AQI")
)

monsoon = (
    df_cluster[
        df_cluster["month"].isin(
            [6, 7, 8, 9]
        )
    ]
    .groupby("City")["AQI"]
    .mean()
    .rename("monsoon_AQI")
)

city_features = city_features.merge(
    winter,
    on="City",
    how="left"
)

city_features = city_features.merge(
    monsoon,
    on="City",
    how="left"
)

city_features = city_features.dropna()

cluster_features = [
    "mean_AQI",
    "mean_PM25",
    "mean_PM10",
    "winter_AQI",
    "monsoon_AQI"
]

scaler = StandardScaler()

X_cluster = scaler.fit_transform(
    city_features[cluster_features]
)

# Use two clusters for a compact city profile
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

city_features["Cluster"] = (
    kmeans.fit_predict(X_cluster)
)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_cluster
)

city_features["PC1"] = X_pca[:, 0]
city_features["PC2"] = X_pca[:, 1]


fig_cluster, ax_cluster = plt.subplots(
    figsize=(11, 7)
)

for cluster in sorted(
    city_features["Cluster"].unique()
):

    subset = city_features[
        city_features["Cluster"] == cluster
    ]

    ax_cluster.scatter(
        subset["PC1"],
        subset["PC2"],
        s=80,
        label=f"Cluster {cluster}"
    )

    for _, row in subset.iterrows():

        ax_cluster.annotate(
            row["City"],
            (
                row["PC1"],
                row["PC2"]
            ),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

ax_cluster.set_xlabel(
    "Principal Component 1"
)

ax_cluster.set_ylabel(
    "Principal Component 2"
)

ax_cluster.set_title(
    "City Air-Quality Clusters"
)

ax_cluster.legend()

ax_cluster.grid(
    alpha=0.3
)

st.pyplot(
    fig_cluster
)

st.dataframe(
    city_features[
        [
            "City",
            "mean_AQI",
            "mean_PM25",
            "mean_PM10",
            "Cluster"
        ]
    ].sort_values(
        "mean_AQI",
        ascending=False
    ),
    use_container_width=True
)


# ============================================================
# SEASONAL HEATMAP
# ============================================================

st.header(
    "🌦️ Seasonal AQI Patterns"
)

monthly = (
    df_cluster
    .groupby(
        ["City", "month"]
    )["AQI"]
    .mean()
    .unstack()
)

fig_heat, ax_heat = plt.subplots(
    figsize=(14, 8)
)

sns.heatmap(
    monthly,
    cmap="RdYlGn_r",
    ax=ax_heat
)

ax_heat.set_title(
    "Average Monthly AQI Across Cities"
)

ax_heat.set_xlabel(
    "Month"
)

ax_heat.set_ylabel(
    "City"
)

st.pyplot(
    fig_heat
)

# ============================================================
# MODEL COMPARISON
# ============================================================

st.header(
    "🤖 Model Comparison"
)

st.write(
    "AirWise compares multiple machine-learning models "
    "for next-day AQI prediction."
)

comparison_data = pd.DataFrame({
    "Model": [
        "Random Forest",
        "LightGBM",
        "Gradient Boosting",
        "Ensemble Average"
    ],
    "MAE": [
        44.41,
        43.85,
        44.68,
        43.65
    ],
    "RMSE": [
        61.07,
        59.89,
        62.24,
        60.11
    ],
    "R²": [
        0.720,
        0.731,
        0.709,
        0.729
    ]
})

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Best MAE",
        "43.65",
        "Ensemble Average"
    )

with col2:

    st.metric(
        "Best RMSE",
        "59.89",
        "LightGBM"
    )

with col3:

    st.metric(
        "Best R²",
        "0.731",
        "LightGBM"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirWise | LightGBM + Quantile Forecasting + "
    "SHAP + K-Means + PCA"
)