"""
Módulo de preprocesamiento para el proyecto de predicción de precios Airbnb.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


ARTIFACTS = {
    "imputer": "notebooks/imputer.pkl",
    "scaler": "notebooks/scaler.pkl",
}

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Aquí iría la lógica de limpieza
    return df


def split_data(df, target="Price", test_size=0.2, random_state=42):
    """Dividir el dataset en entrenamiento y prueba."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_preprocessors(X_train):
    """Ajustar imputador y escalador solo con X_train."""
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    return X_train, imputer, scaler, numeric_cols


def transform_data(X, imputer, scaler, numeric_cols):
    """Aplicar transformaciones guardadas a un nuevo conjunto."""
    X = X.copy()
    X[numeric_cols] = imputer.transform(X[numeric_cols])
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X


def save_artifacts(imputer, scaler, numeric_cols):
    """Guardar objetos entrenados para reutilizar."""
    joblib.dump(imputer, ARTIFACTS["imputer"])
    joblib.dump(scaler, ARTIFACTS["scaler"])
    joblib.dump(list(numeric_cols), "notebooks/numeric_cols.pkl")
