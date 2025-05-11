# Proyecto de ML: Predicción de precios en Airbnb

Este proyecto tiene como objetivo construir un modelo de regresión para predecir precios de alojamientos en Airbnb a partir de datos reales.

## Estructura del proyecto

```
airbnb-pricing-ml/
├── data/
│ ├── raw/ # Datos originales (cargados localmente)
│ └── processed/ # Datos limpios y listos para modelado
├── notebooks/
│ ├── 01_data_exploration.ipynb # EDA, limpieza y anomalías
│ └── 02_model_training.ipynb # Entrenamiento y validación
├── src/
│ ├── preprocessing.py # Funciones de carga y limpieza
│ └── train_model.py # Entrenamiento de modelo
├── models/ # Modelos entrenados (.pkl) 
├── reports/ # Visualizaciones, métricas, resumenes (no lo utilice)
├── .gitignore # Exclusiones estándar
├── Makefile # Comandos automatizados
├── requirements.txt # Dependencias del entorno
└── README.md # Este archivo

```
### 📦 Versiones de librerías utilizadas
```
→ Python        : 3.9.2  
→ pandas        : 1.5.3  
→ numpy         : 1.26.4  
→ scikit-learn  : 1.4.2  
→ matplotlib    : 3.7.5  
→ seaborn       : 0.13.2  
→ plotly        : 5.22.0  
→ scipy         : 1.11.4  
→ statsmodels   : 0.14.2  
→ joblib        : 1.3.2  
```

##  Objetivo

Predecir el precio logarítmico (`Log_Price`) de un alojamiento tipo Airbnb en base a sus características más relevantes, utilizando técnicas modernas de machine learning y análisis exploratorio de datos.

---

##  Flujo de trabajo

### 1. Exploración y preparación de datos (`01. Exploracion...ipynb`)

- Limpieza de columnas con alto porcentaje de valores faltantes
- Transformación de fechas y extracción de variables temporales
- Ingeniería de características: ratios, flags y log transformaciones
- Imputación de valores faltantes
- Detección de outliers mediante métodos estadísticos (Z-score, IQR, Isolation Forest)
- Análisis de colinealidad y reducción mediante VIF

### 2. Modelado y validación (`02. Entrenamiento...ipynb`)

- Comparación de modelos: `Linear Regression`, `Random Forest`, `Gradient Boosting`
- Entrenamiento sobre variable objetivo transformada (`Log_Price`)
- Evaluación con métricas: `R²`, `RMSE`, `MAE`
- Selección final del modelo basado en desempeño
- Exportación del modelo entrenado y variables seleccionadas

### 3. Aplicación interactiva (`streamlit_app.py`) / (en construcción tengo problemas en el deploy con un error de dependencias que no he logrado resolver)

- Formulario dinámico para ingresar las características del alojamiento
```bash
streamlit run notebooks/streamlit_app.py  http://localhost:8501
```
- Carga automática del modelo y variables
- Predicción de `Log_Price` y conversión a escala real (`Price`)
- Botón para descargar resultados en formato CSV

---

## Resultados

Modelo seleccionado: **Random Forest**

| Modelo                  | R²    | MAE    | RMSE   |
|-------------------------|-------|--------|--------|
| Random Forest (raw)     | 0.98  | 0.04   | 0.11   |
| Gradient Boosting (raw) | 0.95  | 0.11   | 0.15   |
| Linear Regression       | 0.74  | 0.27   | 0.35   |

---

##  Cómo ejecutar la app

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

