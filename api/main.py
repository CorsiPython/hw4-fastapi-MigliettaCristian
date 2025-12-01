"""
Homework 4: API di Predizione con FastAPI e Interfaccia Gradio

Questo modulo implementa un'API REST per la predizione dei prezzi delle case
utilizzando FastAPI. L'API carica un modello di machine learning pre-addestrato
e espone un endpoint per effettuare predizioni.

L'API utilizza il dataset California Housing e un modello addestrato sulle
seguenti features:
- MedInc: Reddito mediano nel blocco (in decine di migliaia di dollari)
- HouseAge: Età mediana delle case nel blocco
- AveRooms: Numero medio di stanze per abitazione
- AveBedrms: Numero medio di camere da letto per abitazione
- Population: Popolazione del blocco
- AveOccup: Numero medio di occupanti per abitazione
- Latitude: Latitudine del blocco
- Longitude: Longitudine del blocco

Endpoint da implementare:
- POST /predict: Riceve le features e restituisce la predizione del prezzo

Mantieni le firme e i modelli esattamente come definiti: i test automatici li importano.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path


# Percorso al file del modello (relativo alla cartella api/)
MODEL_PATH = Path(__file__).parent.parent / "model" / "house_price_model.pkl"


class HouseFeatures(BaseModel):
    """Modello Pydantic per le features di input della predizione.
    
    Attributi
    ---------
    MedInc : float
        Reddito mediano nel blocco (in decine di migliaia di dollari)
    HouseAge : float
        Età mediana delle case nel blocco (anni)
    AveRooms : float
        Numero medio di stanze per abitazione
    AveBedrms : float
        Numero medio di camere da letto per abitazione
    Population : float
        Popolazione del blocco
    AveOccup : float
        Numero medio di occupanti per abitazione
    Latitude : float
        Latitudine del blocco
    Longitude : float
        Longitudine del blocco
    """
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictionResponse(BaseModel):
    """Modello Pydantic per la risposta della predizione.
    
    Attributi
    ---------
    predicted_price : float
        Il prezzo predetto della casa (in centinaia di migliaia di dollari)
    """
    predicted_price: float


# Crea l'applicazione FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API per la predizione dei prezzi delle case in California",
    version="1.0.0"
)

# Variabile globale per il modello (caricato all'avvio)
model = None


@app.on_event("startup")
def load_model():
    """Carica il modello all'avvio dell'applicazione."""
    global model
    # TODO: Implementa il caricamento del modello
    # Suggerimento: usa pickle.load() per caricare il modello da MODEL_PATH
    # with open(MODEL_PATH, 'rb') as f:
    #     model = pickle.load(f)
    raise NotImplementedError("Implementa load_model per caricare il modello")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    """Endpoint per la predizione del prezzo di una casa.
    
    Parameters
    ----------
    features : HouseFeatures
        Le caratteristiche della casa per cui predire il prezzo
    
    Returns
    -------
    PredictionResponse
        La risposta con il prezzo predetto
    """
    # TODO: Implementa la predizione
    # 1. Converti features in un array numpy nell'ordine corretto
    # 2. Usa model.predict() per ottenere la predizione
    # 3. Restituisci PredictionResponse con predicted_price
    raise NotImplementedError("Implementa predict per effettuare la predizione")
