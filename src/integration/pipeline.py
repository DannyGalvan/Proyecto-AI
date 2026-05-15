# pipeline.py
"""
Script de integración: datos → ML/DL → NLP → resultado final
"""
from src.data.load_data import load_data
from src.ml.train import train_model
from src.dl.train_dl import train_dl_model
from src.nlp.nlp_component import NLPComponent

# Cargar datos (ajustar según el dominio)
data = load_data()

# Entrenar modelos ML/DL (pueden ser dummy si ya existen modelos entrenados)
ml_model = train_model(data)
dl_model = train_dl_model(data)

# Procesar resultados con NLP
nlp = NLPComponent()
texto = "Este es un ejemplo de texto para análisis NLP."
sentimiento = nlp.sentiment_analysis(texto)

print(f"Sentimiento del texto: {sentimiento}")
# Aquí se puede expandir el pipeline según el flujo real del proyecto
