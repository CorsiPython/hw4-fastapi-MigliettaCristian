"""
Homework 4: Interfaccia Gradio per la Predizione dei Prezzi delle Case

Questo modulo implementa un'interfaccia utente con Gradio per interagire
con l'API FastAPI di predizione dei prezzi delle case.

L'interfaccia deve:
- Fornire componenti di input per tutte le features del modello
- Inviare una richiesta POST all'API quando l'utente clicca "Predict"
- Mostrare il risultato della predizione

Mantieni la struttura come indicata: i test automatici verificheranno
che l'interfaccia sia funzionante.
"""

import gradio as gr
import requests


# URL dell'API FastAPI (deve essere in esecuzione localmente)
API_URL = "http://127.0.0.1:8000/predict"


def predict_price(
    med_inc: float,
    house_age: float,
    ave_rooms: float,
    ave_bedrms: float,
    population: float,
    ave_occup: float,
    latitude: float,
    longitude: float
) -> str:
    """Invia una richiesta all'API e restituisce la predizione.
    
    Parameters
    ----------
    med_inc : float
        Reddito mediano nel blocco
    house_age : float
        Età mediana delle case
    ave_rooms : float
        Numero medio di stanze
    ave_bedrms : float
        Numero medio di camere da letto
    population : float
        Popolazione del blocco
    ave_occup : float
        Numero medio di occupanti
    latitude : float
        Latitudine
    longitude : float
        Longitudine
    
    Returns
    -------
    str
        Messaggio con la predizione o un messaggio di errore
    """
    # TODO: Implementa questa funzione
    # 1. Crea un dizionario con le features
    # 2. Invia una richiesta POST a API_URL con il JSON delle features
    # 3. Gestisci la risposta e restituisci il messaggio appropriato
    # 4. Gestisci eventuali errori di connessione
    
    raise NotImplementedError("Implementa predict_price")


def create_interface() -> gr.Blocks:
    """Crea e restituisce l'interfaccia Gradio.
    
    Returns
    -------
    gr.Blocks
        L'interfaccia Gradio configurata
    """
    # TODO: Implementa questa funzione
    # Suggerimento: usa gr.Blocks() e crea componenti di input come:
    # - gr.Slider per valori con range (es. MedInc, HouseAge)
    # - gr.Number per valori numerici generici
    # - gr.Button per il pulsante "Predict"
    # - gr.Textbox per mostrare il risultato
    
    raise NotImplementedError("Implementa create_interface")


# Entry point per eseguire l'interfaccia
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
