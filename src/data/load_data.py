import pandas as pd

def load_data(file_path):
    """
    Función para cargar y explorar los datos del dataset de transacciones.
    """
    data = pd.read_csv(file_path)  # Asumiendo que el dataset está en formato CSV
    print("Primeras filas del dataset:")
    print(data.head())
    print("\nResumen de estadísticas descriptivas:")
    print(data.describe())
    return data