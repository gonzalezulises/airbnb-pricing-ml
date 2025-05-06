"""
Módulo de preprocesamiento para el proyecto de predicción de precios Airbnb.
"""

import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Aquí iría la lógica de limpieza
    return df
