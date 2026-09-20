import os

import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

from data_prep import load_data, prepare_city_data
from forecast import (
    FEATURES,
    create_features,
    prepare_ml_data,
    chronological_split
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()

delhi_df = prepare_city_data(
    df,
    "Delhi"
)

# ============================================================
# 2. CREATE FEATURES
# ============================================================

feature_df = create_features(
    delhi_df
)

ml_df = prepare_ml_data(
    feature_df
)

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = chronological_split(
    ml_df
)

X_train = train_df[FEATURES]
y_train = train_df["target_AQI"]

X_test = test_df[FEATURES]
y_test = test_df["target_AQI"]


# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)


# ============================================================
# 5. LIGHTGBM
# ============================================================

print("Training LightGBM...")

lgbm = LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

lgbm.fit(
    X_train,
    y_train
)

lgbm_pred = lgbm.predict(
    X_test
)


# ============================================================
# 6. GRADIENT BOOSTING
# ============================================================

print("Training Gradient Boosting...")

gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

gb_pred = gb.predict(
    X_test
)


# ============================================================
# 7. ENSEMBLE
# ============================================================

ensemble_pred = (
    rf_pred +
    lgbm_pred +
    gb_pred
) / 3


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    y_true,
    prediction
):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    rmse = mean_squared_error(
        y_true,
        prediction
    ) ** 0.5

    r2 = r2_score(
        y_true,
        prediction
    )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results = [

    evaluate(
        "Random Forest",
        y_test,
        rf_pred
    ),

    evaluate(
        "LightGBM",
        y_test,
        lgbm_pred
    ),

    evaluate(
        "Gradient Boosting",
        y_test,
        gb_pred
    ),

    evaluate(
        "Ensemble Average",
        y_test,
        ensemble_pred
    )
]

results_df = pd.DataFrame(
    results
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print(
    "\n========== MODEL COMPARISON =========="
)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.3f}".format
        }
    )
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs(
    "outputs/metrics",
    exist_ok=True
)

results_df.to_csv(
    "outputs/metrics/model_comparison.csv",
    index=False
)

print(
    "\nSaved:"
    " outputs/metrics/model_comparison.csv"
)