import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Calcular rutas absolutas desde la ubicación del script
base_path = Path(__file__).resolve().parent
model_path = base_path.parent / "models" / "final_model.pkl"
vars_path = base_path.parent / "models" / "vars_seleccionadas.txt"

# --- Cargar modelo entrenado ---
try:
    st_model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"No se encontró el modelo en: {model_path}")
    st.stop()

# --- Cargar columnas usadas en entrenamiento ---
try:
    with open(vars_path, "r") as f:
        columnas_modelo = f.read().splitlines()
except FileNotFoundError:
    st.error(f"No se encontró el archivo de columnas: {vars_path}")
    st.stop()

# --- UI de la app ---
st.title("Práctica de predicción de precios ML101")

if "ejemplo" not in st.session_state:
    st.session_state.ejemplo = pd.DataFrame({col: [0.0] for col in columnas_modelo})

st.write("Ingresa los datos del alojamiento para predecir el precio:")
df_input = st.data_editor(st.session_state.ejemplo, num_rows="dynamic")

# Validar columnas necesarias
missing_cols = [col for col in columnas_modelo if col not in df_input.columns]
if missing_cols:
    st.error(f"Faltan columnas requeridas por el modelo: {missing_cols}")
else:
    if st.button("Predecir precio"):
        try:
            y_pred_log = st_model.predict(df_input[columnas_modelo])
            y_pred = np.expm1(y_pred_log)

            st.success("Predicción completada")
            st.write(f"Precio estimado: ${y_pred[0]:,.2f}")

            # Para descarga
            df_output = df_input.copy()
            df_output["Log_Price_Predicho"] = y_pred_log
            df_output["Price_Predicho"] = y_pred

            csv = df_output.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar resultados", csv, "predicciones.csv", "text/csv")
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# Firma
st.markdown("---")
st.markdown("Aplicación desarrollada por **Ulises González**, Panamá 2025")