import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_prep import load_data


# ==========================================
# 1. LOAD DATA
# ==========================================

df = load_data()

# ==========================================
# 2. CREATE MONTH
# ==========================================

df["month"] = df["Date"].dt.month

# ==========================================
# 3. CITY-LEVEL FEATURES
# ==========================================

city_features = (
    df.groupby("City")
    .agg(
        mean_AQI=("AQI", "mean"),
        mean_PM25=("PM2.5", "mean"),
        mean_PM10=("PM10", "mean"),
    )
    .reset_index()
)

# ==========================================
# 4. SEASONAL FEATURES
# ==========================================

winter = (
    df[df["month"].isin([11, 12, 1, 2])]
    .groupby("City")["AQI"]
    .mean()
    .rename("winter_AQI")
)

monsoon = (
    df[df["month"].isin([6, 7, 8, 9])]
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

print("\n========== CITY FEATURES ==========")

print(city_features)

# ==========================================
# 5. SCALE
# ==========================================

feature_columns = [
    "mean_AQI",
    "mean_PM25",
    "mean_PM10",
    "winter_AQI",
    "monsoon_AQI",
]

X = city_features[feature_columns]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# 6. FIND BEST K USING SILHOUETTE
# ==========================================

print("\n========== SILHOUETTE SCORES ==========")

best_k = None
best_score = -1

for k in range(2, 6):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K={k} -> silhouette={score:.3f}"
    )

    if score > best_score:
        best_score = score
        best_k = k

print(
    f"\nSelected K = {best_k}"
)

# ==========================================
# 7. FINAL K-MEANS
# ==========================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

city_features["Cluster"] = (
    kmeans.fit_predict(X_scaled)
)

# ==========================================
# 8. PCA
# ==========================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

city_features["PC1"] = X_pca[:, 0]

city_features["PC2"] = X_pca[:, 1]

# ==========================================
# 9. PLOT
# ==========================================

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

plt.figure(figsize=(10, 7))

for cluster in sorted(
    city_features["Cluster"].unique()
):

    subset = city_features[
        city_features["Cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster}",
        s=80
    )

    for _, row in subset.iterrows():

        plt.annotate(
            row["City"],
            (
                row["PC1"],
                row["PC2"]
            ),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title(
    "AirWise: City Air-Quality Clusters"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/figures/city_clusters.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 10. SAVE DATA
# ==========================================

city_features.to_csv(
    "outputs/metrics/city_clusters.csv",
    index=False
)

print(
    "\nCluster plot saved to "
    "outputs/figures/city_clusters.png"
)

print(
    "Cluster data saved to "
    "outputs/metrics/city_clusters.csv"
)