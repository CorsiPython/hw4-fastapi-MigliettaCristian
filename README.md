[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D90FY3ls)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21899040&assignment_repo_type=AssignmentRepo)
# Homework 4: API di Predizione con FastAPI e Interfaccia Gradio

Questo repository è il punto di partenza per l'homework "API di Predizione".
L'obiettivo è creare un sistema completo per la predizione dei prezzi delle case,
composto da un backend API (FastAPI) e un frontend interattivo (Gradio).

In questa versione "starter" trovi lo scheletro del codice da completare e un
setup di test basato su `pytest`.

---

## Architettura del Progetto

```
┌─────────────────┐      POST /predict      ┌─────────────────┐
│                 │  ──────────────────────▶│                 │
│  Gradio UI      │                         │  FastAPI        │
│  (app.py)       │◀──────────────────────  │  (api/main.py)  │
│                 │    {"predicted_price"}  │                 │
└─────────────────┘                         └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │   ML Model      │
                                            │   (pickle)      │
                                            └─────────────────┘
```

---

## Dataset e Modello

Il progetto utilizza il **California Housing Dataset** e un modello di regressione
pre-addestrato. Il modello è salvato in `model/house_price_model.pkl`.

### Features del Modello

| Feature | Descrizione |
|---------|-------------|
| `MedInc` | Reddito mediano nel blocco (×10k $) |
| `HouseAge` | Età mediana delle case (anni) |
| `AveRooms` | Numero medio di stanze per abitazione |
| `AveBedrms` | Numero medio di camere da letto |
| `Population` | Popolazione del blocco |
| `AveOccup` | Numero medio di occupanti |
| `Latitude` | Latitudine |
| `Longitude` | Longitudine |

### Output

Il modello predice `MedHouseVal`: valore mediano delle case in centinaia di migliaia di dollari.
(es. 2.5 = $250,000)

---

## Cosa devi fare

### Parte 1: Backend API (`api/main.py`)

1. **Crea l'applicazione FastAPI**
   ```python
   app = FastAPI(title="House Price Prediction API")
   ```

2. **Carica il modello all'avvio**
   ```python
   with open(MODEL_PATH, 'rb') as f:
       model = pickle.load(f)
   ```

3. **Implementa l'endpoint POST `/predict`**
   - Riceve un JSON con le features (modello `HouseFeatures`)
   - Usa il modello per calcolare la predizione
   - Restituisce `{"predicted_price": <valore>}`

### Parte 2: Frontend Gradio (`app.py`)

1. **Implementa `predict_price()`**
   - Raccoglie i valori di input
   - Invia una richiesta POST all'API
   - Restituisce il risultato o un messaggio di errore

2. **Implementa `create_interface()`**
   - Crea componenti di input per ogni feature
   - Aggiunge un pulsante "Predict"
   - Mostra il risultato in un componente di output

---

## Struttura del repository

```
hw4-fastapi-gradio/
├── api/
│   ├── __init__.py
│   └── main.py                    # API FastAPI (DA COMPLETARE)
├── model/
│   └── house_price_model.pkl      # Modello pre-addestrato
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures pytest
│   └── test_api_public.py         # Test pubblici
├── app.py                         # Interfaccia Gradio (DA COMPLETARE)
├── create_model.py                # Script per creare il modello
├── requirements.txt               # Dipendenze
├── pytest.ini                     # Configurazione pytest
└── README.md
```

---

## Setup locale (consigliato)

1) Naviga nella directory del progetto:

```bash
cd hw4-fastapi-gradio
```

2) Crea ed attiva un ambiente virtuale con `uv`:

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# oppure .venv\Scripts\activate su Windows
```

3) Installa le dipendenze:

```bash
uv pip install -r requirements.txt
```

4) Crea il modello (solo la prima volta):

```bash
uv run python create_model.py
```

5) Esegui i test:

```bash
uv run pytest
```

---

## Esecuzione dell'applicazione

### Avvia l'API FastAPI

```bash
uv run uvicorn api.main:app --reload
```

L'API sarà disponibile su http://127.0.0.1:8000

### Avvia l'interfaccia Gradio (in un altro terminale)

```bash
uv run python app.py
```

L'interfaccia sarà disponibile su http://127.0.0.1:7860

---

## Suggerimenti per l'implementazione

### Endpoint FastAPI

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    # Converti le features in array numpy
    X = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]])
    
    # Effettua la predizione
    prediction = model.predict(X)[0]
    
    return PredictionResponse(predicted_price=float(prediction))
```

### Interfaccia Gradio

```python
def create_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# House Price Predictor")
        
        with gr.Row():
            med_inc = gr.Slider(0, 15, value=3.5, label="Median Income")
            house_age = gr.Slider(1, 52, value=25, label="House Age")
        
        # ... altri input ...
        
        predict_btn = gr.Button("Predict")
        output = gr.Textbox(label="Predicted Price")
        
        predict_btn.click(
            fn=predict_price,
            inputs=[med_inc, house_age, ...],
            outputs=output
        )
    
    return demo
```

### Richiesta HTTP con requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"MedInc": 3.5, "HouseAge": 25, ...}
)

if response.status_code == 200:
    data = response.json()
    price = data["predicted_price"]
```

---

## Valutazione

Il tuo codice sarà valutato automaticamente tramite i test pytest forniti.

I test verificano:
- ✅ Modelli Pydantic corretti (`HouseFeatures`, `PredictionResponse`)
- ✅ App FastAPI funzionante
- ✅ Endpoint `/predict` che restituisce JSON valido
- ✅ Predizioni consistenti con il modello
- ✅ Gestione errori per input invalidi
- ✅ Struttura base dell'interfaccia Gradio

**Nota**: L'interfaccia Gradio sarà valutata manualmente durante la presentazione.

---

## Test

Per eseguire tutti i test:

```bash
uv run pytest
```

Per eseguire test specifici:

```bash
# Solo test dell'API
uv run pytest tests/test_api_public.py::TestPredictEndpoint

# Solo test dei modelli Pydantic
uv run pytest tests/test_api_public.py::TestPydanticModels
```

---

## Note importanti

- Assicurati che il modello `house_price_model.pkl` esista prima di avviare l'API
- L'API deve essere in esecuzione (porta 8000) per testare l'interfaccia Gradio
- I valori predetti sono in centinaia di migliaia di dollari
- Gestisci sempre gli errori di connessione nell'interfaccia Gradio
