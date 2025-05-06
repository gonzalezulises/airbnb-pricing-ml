"""
Script de entrenamiento del modelo.
"""

from src.preprocessing import load_data, clean_data

def main():
    df = load_data("data/processed/airbnb_clean.csv")
    df = clean_data(df)
    # Aquí iría el pipeline de entrenamiento

if __name__ == "__main__":
    main()
