# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo entrenado
model_path = "../models/final_model_stacked.pkl"
st_model = joblib.load(model_path)

# Cargar columnas usadas en entrenamiento
vars_path = "../models/vars_seleccionadas.txt"
with open(vars_path) as f:
    columnas_modelo = f.read().splitlines()

# Título y entrada interactiva
st.title("Predicción de precios con modelo Stacking")

if "ejemplo" not in st.session_state:
    st.session_state.ejemplo = pd.DataFrame({col: [0.0] for col in columnas_modelo})

st.write("### Ingresa los datos del nuevo alojamiento para predecir el precio:")
df_input = st.data_editor(st.session_state.ejemplo, num_rows="dynamic")

# Validar columnas
missing_cols = [col for col in columnas_modelo if col not in df_input.columns]
if missing_cols:
    st.error(f"Faltan columnas: {missing_cols}")
else:
    if st.button("Predecir precio"):
        y_pred_log = st_model.predict(df_input[columnas_modelo])
        y_pred = np.expm1(y_pred_log)

        df_output = df_input.copy()
        df_output["Price_log_predicho"] = y_pred_log
        df_output["Price_predicho"] = y_pred

        st.success("✅ Predicción completada")
        st.write(df_output)

        # Descargar resultados
        csv = df_output.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados", csv, "predicciones.csv", "text/csv")
