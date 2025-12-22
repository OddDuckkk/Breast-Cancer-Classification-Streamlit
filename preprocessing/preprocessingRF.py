import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
def preprocess_rf(df: pd.DataFrame):

    # ===== PEMBAGIAN LABEL & FITUR =====
    #Label
    X = df.loc[:, FITUR]
    #Fitur
    y = df.diagnosis

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le, scaler
