# validation.py
import pandas as pd

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

REQUIRED_COLUMNS = [
    "diagnosis",
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

def validate_dataset(df: pd.DataFrame):
    dataset_columns = df.columns.tolist()

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in dataset_columns
    ]

    extra_columns = [
        col for col in dataset_columns
        if col not in REQUIRED_COLUMNS
    ]

    is_valid = len(missing_columns) == 0

    return is_valid, missing_columns, extra_columns

def validate_data_klasifikasi(df: pd.DataFrame):
    dataset_columns = df.columns.tolist()

    missing_columns = [
        col for col in FITUR
        if col not in dataset_columns
    ]

    extra_columns = [
        col for col in dataset_columns
        if col not in FITUR
    ]

    is_valid = len(missing_columns) == 0

    return is_valid, missing_columns, extra_columns
