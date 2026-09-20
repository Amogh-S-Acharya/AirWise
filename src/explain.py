import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from data_prep import load_data, prepare_city_data
from forecast import (
    FEATURES,
    create_features,
    prepare_ml_data,
    chronological_split,
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = load_data()

delhi_df = prepare_city_data(
    df,
    "Delhi"
)

# ==========================================
# 2. CREATE FEATURES
# ==========================================

feature_df = create_features(
    delhi_df
)

ml_df = prepare_ml_data(
    feature_df
)

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

train_df, test_df = chronological_split(
    ml_df
)

X_train = train_df[FEATURES]

y_train = train_df["target_AQI"]

X_test = test_df[FEATURES]

# ==========================================
# 4. TRAIN MEDIAN MODEL
# ==========================================

model = LGBMRegressor(
    objective="quantile",
    alpha=0.50,
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

# ==========================================
# 5. SHAP
# ==========================================

print("\n========== CALCULATING SHAP ==========")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer(
    X_test
)

# ==========================================
# 6. OUTPUT DIRECTORY
# ==========================================

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

# ==========================================
# 7. GLOBAL SHAP SUMMARY
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP summary saved to "
    "outputs/figures/shap_summary.png"
)

# ==========================================
# 8. SHAP BAR IMPORTANCE
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/shap_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP importance saved to "
    "outputs/figures/shap_importance.png"
)