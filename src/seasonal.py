import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import load_data


df = load_data()

df["month"] = df["Date"].dt.month

monthly = (
    df.groupby(["City", "month"])["AQI"]
    .mean()
    .unstack()
)

plt.figure(figsize=(14, 9))

sns.heatmap(
    monthly,
    cmap="RdYlGn_r",
    annot=False
)

plt.title(
    "Average Monthly AQI Across Cities"
)

plt.xlabel("Month")

plt.ylabel("City")

plt.tight_layout()

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

plt.savefig(
    "outputs/figures/seasonal_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: outputs/figures/seasonal_heatmap.png"
)