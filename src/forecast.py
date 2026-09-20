import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

from data_prep import load_data, prepare_city_data


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


def create_features(df):

    df = df.copy()

    df = df.sort_values("Date").reset_index(drop=True)

    # Historical AQI features
    df["AQI_lag_1"] = df["AQI"].shift(1)

    df["AQI_lag_7"] = df["AQI"].shift(7)

    df["AQI_rolling_7"] = (
        df["AQI"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Historical pollutant features
    df["PM2.5_lag_1"] = df["PM2.5"].shift(1)

    df["PM10_lag_1"] = df["PM10"].shift(1)

    df["NO2_lag_1"] = df["NO2"].shift(1)

    df["CO_lag_1"] = df["CO"].shift(1)

    # Calendar features
    df["month"] = df["Date"].dt.month

    df["day_of_week"] = df["Date"].dt.dayofweek

    # Tomorrow's AQI
    df["target_AQI"] = df["AQI"].shift(-1)

    return df


def prepare_ml_data(df):

    ml_df = df[
        ["Date"] + FEATURES + ["target_AQI"]
    ].copy()

    ml_df = ml_df.dropna()

    return ml_df


def chronological_split(df, train_ratio=0.8):

    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def train_quantile_model(X_train, y_train, alpha):

    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    df = load_data()

    # ==========================================
    # 2. SELECT DELHI
    # ==========================================

    delhi_df = prepare_city_data(df, "Delhi")

    # ==========================================
    # 3. CREATE FEATURES
    # ==========================================

    feature_df = create_features(delhi_df)

    # ==========================================
    # 4. PREPARE ML DATA
    # ==========================================

    ml_df = prepare_ml_data(feature_df)

    print("\n========== ML DATA ==========")
    print("Rows:", len(ml_df))

    # ==========================================
    # 5. CHRONOLOGICAL SPLIT
    # ==========================================

    train_df, test_df = chronological_split(ml_df)

    X_train = train_df[FEATURES]
    y_train = train_df["target_AQI"]

    X_test = test_df[FEATURES]
    y_test = test_df["target_AQI"]

    print("\n========== TRAIN / TEST ==========")

    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("\nTraining period:")
    print(train_df["Date"].min(), "to", train_df["Date"].max())

    print("\nTesting period:")
    print(test_df["Date"].min(), "to", test_df["Date"].max())

    # ==========================================
    # 6. TRAIN P10, P50, P90
    # ==========================================

    print("\n========== TRAINING QUANTILE MODELS ==========")

    model_p10 = train_quantile_model(
        X_train,
        y_train,
        0.10
    )

    model_p50 = train_quantile_model(
        X_train,
        y_train,
        0.50
    )

    model_p90 = train_quantile_model(
        X_train,
        y_train,
        0.90
    )

    # ==========================================
    # 7. PREDICTIONS
    # ==========================================

    pred_p10 = model_p10.predict(X_test)

    pred_p50 = model_p50.predict(X_test)

    pred_p90 = model_p90.predict(X_test)

    # ==========================================
    # 8. ENSURE INTERVAL ORDER
    # ==========================================

    lower = np.minimum(pred_p10, pred_p90)
    upper = np.maximum(pred_p10, pred_p90)

    # ==========================================
    # 9. MEDIAN MODEL METRICS
    # ==========================================

    mae = mean_absolute_error(
        y_test,
        pred_p50
    )

    rmse = mean_squared_error(
        y_test,
        pred_p50
    ) ** 0.5

    r2 = r2_score(
        y_test,
        pred_p50
    )

    # ==========================================
    # 10. COVERAGE
    # ==========================================

    coverage = (
        (y_test >= lower) &
        (y_test <= upper)
    ).mean()

    average_interval_width = (
        upper - lower
    ).mean()

    # ==========================================
    # 11. PRINT RESULTS
    # ==========================================

    print("\n========== AIRWISE RESULTS ==========")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    print("\n========== UNCERTAINTY ==========")

    print(
        f"Empirical P10-P90 coverage: "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"Average interval width: "
        f"{average_interval_width:.2f} AQI points"
    )

    # ==========================================
    # 12. SAVE OUTPUT DIRECTORY
    # ==========================================

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    os.makedirs(
        "outputs/metrics",
        exist_ok=True
    )

    # ==========================================
    # 13. CREATE FORECAST RESULT TABLE
    # ==========================================

    results = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_AQI": y_test.values,
        "P10": lower,
        "P50": pred_p50,
        "P90": upper
    })

    results.to_csv(
        "outputs/metrics/forecast_results.csv",
        index=False
    )

    # ==========================================
    # 14. FORECAST VISUALIZATION
    # ==========================================

    plt.figure(figsize=(14, 6))

    plt.plot(
        results["Date"],
        results["Actual_AQI"],
        label="Actual AQI",
        linewidth=1.5
    )

    plt.plot(
        results["Date"],
        results["P50"],
        label="Median Forecast (P50)",
        linewidth=1.5
    )

    plt.fill_between(
        results["Date"],
        results["P10"],
        results["P90"],
        alpha=0.2,
        label="P10-P90 Prediction Interval"
    )

    plt.title(
        "AirWise: Delhi Next-Day AQI Forecast with Uncertainty"
    )

    plt.xlabel("Date")
    plt.ylabel("AQI")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/aqi_forecast_uncertainty.png",
        dpi=200
    )

    plt.close()

    print(
        "\nForecast graph saved to:"
        " outputs/figures/aqi_forecast_uncertainty.png"
    )

    print(
        "Forecast results saved to:"
        " outputs/metrics/forecast_results.csv"
    )