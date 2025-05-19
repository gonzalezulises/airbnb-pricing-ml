"""
Script de entrenamiento del modelo.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.preprocessing import (
    load_data,
    clean_data,
    split_data,
    fit_preprocessors,
    transform_data,
    save_artifacts,
)

def main():
    df = load_data("data/raw/airbnb-listings-extract.csv")
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df, target="Price")
    X_train, imputer, scaler, numeric_cols = fit_preprocessors(X_train)
    X_test = transform_data(X_test, imputer, scaler, numeric_cols)

    save_artifacts(imputer, scaler, numeric_cols)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

    joblib.dump(model, "notebooks/final_model.pkl")

if __name__ == "__main__":
    main()
