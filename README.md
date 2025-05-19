# 📘 Comparación de Modelos de Regresión para Predicción de Precios en Airbnb

## 📌 Descripción general

Este notebook implementa un pipeline completo de **entrenamiento, evaluación y comparación de modelos de regresión supervisada** aplicado a un dataset tipo Airbnb, con el objetivo de predecir precios de propiedades.

Se emplean técnicas de **preprocesamiento, escalado, imputación, codificación categórica y validación cruzada** con ajuste de hiperparámetros vía `GridSearchCV`.

---

## 🧠 Modelos implementados

- **Lasso Regression**
- **Ridge Regression**
- **ElasticNet**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

---

## ⚙️ Flujo de trabajo

1. **Carga y limpieza de datos**
   - Eliminación de columnas irrelevantes  
   - Imputación de valores nulos  
   - Conversión de tipos y creación de nuevas variables

2. **Preprocesamiento**
   - Escalado de variables numéricas  
   - Codificación de variables categóricas  
   - División de datos en entrenamiento y prueba (`train_test_split`)

3. **Entrenamiento de modelos**
   - Uso de `GridSearchCV` con validación cruzada (`cv=5`)  
   - Comparación de métricas: `MSE`, `MAE`, `R²`, `Median AE`, tiempo de ejecución  
   - Selección del mejor modelo

4. **Análisis del mejor modelo**
   - Importancia de variables (`feature_importances_` o `coef_`)  
   - Gráfico de predicción vs realidad  
   - Histograma de errores de predicción

---

## 📊 Métricas clave (ejemplo de resultados obtenidos)

| Modelo             | Test MSE | Test R² | MAE  | Tiempo (s) |
|--------------------|----------|---------|------|------------|
| Gradient Boosting  | **374.76** | **0.735** | 13.31 | 20.01       |
| Random Forest      | 390.98   | 0.724   | 13.51 | 91.27       |
| Lasso / Ridge / ElasticNet | ~529 | ~0.626 | ~16.2 | <1 s        |

---

## 🖼️ Visualizaciones

- Comparación gráfica de modelos (`R²`, `MSE`, `MAE`, tiempo)
- Top 10 variables más importantes
- Predicción vs realidad
- Distribución de errores de predicción

---

## 📁 Requisitos

- Python 3.8+
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- category_encoders  

Instalación rápida:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn category_encoders
