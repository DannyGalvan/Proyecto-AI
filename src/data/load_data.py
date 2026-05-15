import os
import sys
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import DATASET_PATH


DEFAULT_DATASET_PATH = str(DATASET_PATH)


def load_data(file_path: str | None = None):
    """
    Función para cargar y explorar los datos del dataset de transacciones.
    """
    dataset_path = file_path or DEFAULT_DATASET_PATH
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"No se encontró el dataset en: {dataset_path}. "
            "Coloca el archivo en data/financial_data.csv"
        )

    data = pd.read_csv(dataset_path)
    print("Primeras filas del dataset:")
    print(data.head())
    print("\nResumen de estadísticas descriptivas:")
    print(data.describe())
    return data