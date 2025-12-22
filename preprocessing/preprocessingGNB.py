import pandas as pd
import numpy as np

FITUR = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# =========================
# FUNGSI PREPROCESS GAUSSIAN NAIVE BAYES
# =========================
def preprocess_gnb(df: pd.DataFrame):

    # ===== PEMBAGIAN LABEL & FITUR =====
    #Label
    X = df.loc[:, FITUR]
    #Fitur
    Y = df.diagnosis

    # ===== SELEKSI FITUR BERDASARKAN KORELASI =====
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold
    threshold = 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    X = X.drop(columns = to_drop)
    return X, Y, to_drop