# Proyecto de ML: Predicción de precios en Airbnb

Este proyecto tiene como objetivo construir un modelo de regresión para predecir precios de alojamientos en Airbnb a partir de datos reales.

## Estructura del proyecto

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
├── reports/ # Visualizaciones, métricas, resumenes
├── .gitignore # Exclusiones estándar
├── Makefile # Comandos automatizados
├── requirements.txt # Dependencias del entorno
└── README.md # Este archivo


## Requerimientos del entorno

Proyecto ejecutado y probado con Python 3.12 y las siguientes dependencias clave:

numpy==1.23.*
pandas==1.5.*
scipy==1.9.*
matplotlib==3.6.*
seaborn==0.12.*
plotly==5.13.*
scikit-learn==1.1.3
jupyterlab==3.4.*
shap==0.41.*
xgboost==1.7.*
lightgbm==4.6.0
ruff==0.0.270

Procedimiento realizado hasta ahora

- Inicialización del proyecto

  -Creación desde cero con entorno virtual (venv)

  -Proyecto versionado en Git y sincronizado con GitHub

  -Configuración de .gitignore para excluir archivos innecesarios

- Instalación de dependencias

  -Se instaló requirements.txt con librerías científicas, ML y herramientas de estilo

  -Se resolvieron problemas de compilación de lightgbm mediante brew install cmake libomp (en macOS)

- Estructura base generada

  -Carpetas para datos, scripts, modelos y notebooks

  -Makefile para automatizar comandos: install, notebook, train, lint, clean

- Incorporación de datos

  -Dataset real cargado manualmente en data/raw/

  -Confirmado que el tamaño (60 MB) es apto para subirlo al repositorio de github
