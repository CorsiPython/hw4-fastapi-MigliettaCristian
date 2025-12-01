"""
Test pubblici per l'Homework 4: API FastAPI e Interfaccia Gradio

Questi test verificano le funzionalità dell'API FastAPI.
Gli studenti devono implementare il codice in modo che tutti i test passino.
"""

import pytest
import pickle
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Aggiungi la directory parent al path per importare i moduli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test per i modelli Pydantic
# =============================================================================

class TestPydanticModels:
    """Test per i modelli Pydantic HouseFeatures e PredictionResponse."""
    
    def test_house_features_import(self):
        """Verifica che HouseFeatures sia importabile."""
        from api.main import HouseFeatures
        assert HouseFeatures is not None
    
    def test_prediction_response_import(self):
        """Verifica che PredictionResponse sia importabile."""
        from api.main import PredictionResponse
        assert PredictionResponse is not None
    
    def test_house_features_creation(self, sample_features):
        """Verifica che HouseFeatures possa essere creato con i dati corretti."""
        from api.main import HouseFeatures
        
        features = HouseFeatures(**sample_features)
        
        assert features.MedInc == sample_features["MedInc"]
        assert features.HouseAge == sample_features["HouseAge"]
        assert features.AveRooms == sample_features["AveRooms"]
        assert features.AveBedrms == sample_features["AveBedrms"]
        assert features.Population == sample_features["Population"]
        assert features.AveOccup == sample_features["AveOccup"]
        assert features.Latitude == sample_features["Latitude"]
        assert features.Longitude == sample_features["Longitude"]
    
    def test_house_features_validation_error(self):
        """Verifica che HouseFeatures sollevi errore con dati mancanti."""
        from api.main import HouseFeatures
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            HouseFeatures(MedInc=3.5)  # Mancano gli altri campi
    
    def test_house_features_type_coercion(self):
        """Verifica che HouseFeatures converta i tipi correttamente."""
        from api.main import HouseFeatures
        
        # Passa interi invece di float
        features = HouseFeatures(
            MedInc=3,
            HouseAge=25,
            AveRooms=5,
            AveBedrms=1,
            Population=1200,
            AveOccup=3,
            Latitude=37,
            Longitude=-122
        )
        
        # Pydantic dovrebbe convertire a float
        assert isinstance(features.MedInc, (int, float))
    
    def test_prediction_response_creation(self):
        """Verifica che PredictionResponse possa essere creato."""
        from api.main import PredictionResponse
        
        response = PredictionResponse(predicted_price=2.5)
        assert response.predicted_price == 2.5
    
    def test_prediction_response_validation_error(self):
        """Verifica che PredictionResponse sollevi errore senza predicted_price."""
        from api.main import PredictionResponse
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            PredictionResponse()


# =============================================================================
# Test per l'applicazione FastAPI
# =============================================================================

class TestFastAPIApp:
    """Test per l'applicazione FastAPI."""
    
    def test_app_import(self, model_path):
        """Verifica che l'app FastAPI sia importabile."""
        # Patch MODEL_PATH per usare il modello di test
        with patch('api.main.MODEL_PATH', model_path):
            # Rimuovi il modulo dalla cache per forzare il reimport
            if 'api.main' in sys.modules:
                del sys.modules['api.main']
            
            from api.main import app
            assert app is not None
    
    def test_app_is_fastapi_instance(self, model_path):
        """Verifica che app sia un'istanza di FastAPI."""
        from fastapi import FastAPI
        
        with patch('api.main.MODEL_PATH', model_path):
            if 'api.main' in sys.modules:
                del sys.modules['api.main']
            
            from api.main import app
            assert isinstance(app, FastAPI)


# =============================================================================
# Test per l'endpoint /predict con TestClient
# =============================================================================

class TestPredictEndpoint:
    """Test per l'endpoint POST /predict."""
    
    @pytest.fixture(autouse=True)
    def setup_app(self, model_path):
        """Setup dell'app con il modello di test."""
        with patch('api.main.MODEL_PATH', model_path):
            if 'api.main' in sys.modules:
                del sys.modules['api.main']
            
            from api.main import app
            from fastapi.testclient import TestClient
            
            self.client = TestClient(app)
            self.app = app
    
    def test_predict_endpoint_exists(self):
        """Verifica che l'endpoint /predict esista."""
        response = self.client.post("/predict", json={
            "MedInc": 3.5,
            "HouseAge": 25.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.1,
            "Population": 1200.0,
            "AveOccup": 3.0,
            "Latitude": 37.5,
            "Longitude": -122.0
        })
        
        # Non dovrebbe essere 404 (Not Found) o 405 (Method Not Allowed)
        assert response.status_code != 404, "L'endpoint /predict non esiste"
        assert response.status_code != 405, "L'endpoint /predict non accetta POST"
    
    def test_predict_returns_200(self, sample_features):
        """Verifica che /predict restituisca status 200."""
        response = self.client.post("/predict", json=sample_features)
        assert response.status_code == 200, \
            f"Atteso status 200, ricevuto {response.status_code}"
    
    def test_predict_returns_json(self, sample_features):
        """Verifica che /predict restituisca JSON."""
        response = self.client.post("/predict", json=sample_features)
        
        # Verifica che il content-type sia JSON
        assert "application/json" in response.headers.get("content-type", ""), \
            "La risposta deve essere JSON"
    
    def test_predict_response_has_predicted_price(self, sample_features):
        """Verifica che la risposta contenga 'predicted_price'."""
        response = self.client.post("/predict", json=sample_features)
        data = response.json()
        
        assert "predicted_price" in data, \
            "La risposta deve contenere 'predicted_price'"
    
    def test_predict_price_is_number(self, sample_features):
        """Verifica che predicted_price sia un numero."""
        response = self.client.post("/predict", json=sample_features)
        data = response.json()
        
        assert isinstance(data["predicted_price"], (int, float)), \
            "predicted_price deve essere un numero"
    
    def test_predict_price_is_positive(self, sample_features):
        """Verifica che predicted_price sia positivo (prezzi non negativi)."""
        response = self.client.post("/predict", json=sample_features)
        data = response.json()
        
        # Il prezzo può essere basso ma non dovrebbe essere negativo
        # (nota: alcuni modelli potrebbero predire valori negativi per input estremi)
        assert data["predicted_price"] > -10, \
            "predicted_price sembra troppo negativo"
    
    def test_predict_with_different_inputs(self):
        """Verifica che predizioni diverse diano risultati diversi."""
        # Input con reddito basso
        low_income = {
            "MedInc": 1.0,
            "HouseAge": 40.0,
            "AveRooms": 4.0,
            "AveBedrms": 1.0,
            "Population": 2000.0,
            "AveOccup": 4.0,
            "Latitude": 34.0,
            "Longitude": -118.0
        }
        
        # Input con reddito alto
        high_income = {
            "MedInc": 10.0,
            "HouseAge": 10.0,
            "AveRooms": 8.0,
            "AveBedrms": 1.5,
            "Population": 500.0,
            "AveOccup": 2.0,
            "Latitude": 37.5,
            "Longitude": -122.0
        }
        
        response_low = self.client.post("/predict", json=low_income)
        response_high = self.client.post("/predict", json=high_income)
        
        price_low = response_low.json()["predicted_price"]
        price_high = response_high.json()["predicted_price"]
        
        # Il prezzo per reddito alto dovrebbe essere maggiore
        assert price_high > price_low, \
            "Case con reddito mediano più alto dovrebbero avere prezzi più alti"
    
    def test_predict_validation_error(self):
        """Verifica che dati mancanti restituiscano errore 422."""
        response = self.client.post("/predict", json={"MedInc": 3.5})
        
        assert response.status_code == 422, \
            "Dati incompleti dovrebbero restituire status 422 (Validation Error)"
    
    def test_predict_invalid_json(self):
        """Verifica che JSON invalido restituisca errore."""
        response = self.client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422], \
            "JSON invalido dovrebbe restituire errore"


# =============================================================================
# Test per la consistenza delle predizioni
# =============================================================================

class TestPredictionConsistency:
    """Test per verificare la consistenza delle predizioni."""
    
    @pytest.fixture(autouse=True)
    def setup_app(self, model_path, trained_model):
        """Setup dell'app con il modello di test."""
        with patch('api.main.MODEL_PATH', model_path):
            if 'api.main' in sys.modules:
                del sys.modules['api.main']
            
            from api.main import app
            from fastapi.testclient import TestClient
            
            self.client = TestClient(app)
            self.trained_model = trained_model
    
    def test_prediction_matches_model(self, sample_features, sample_features_list):
        """Verifica che la predizione dell'API corrisponda al modello diretto."""
        # Predizione tramite API
        response = self.client.post("/predict", json=sample_features)
        api_prediction = response.json()["predicted_price"]
        
        # Predizione diretta dal modello
        X = np.array([sample_features_list])
        model_prediction = self.trained_model.predict(X)[0]
        
        # Le predizioni dovrebbero essere molto simili
        assert np.isclose(api_prediction, model_prediction, rtol=0.01), \
            f"API: {api_prediction}, Modello: {model_prediction}"
    
    def test_same_input_same_output(self, sample_features):
        """Verifica che lo stesso input dia sempre lo stesso output."""
        response1 = self.client.post("/predict", json=sample_features)
        response2 = self.client.post("/predict", json=sample_features)
        
        price1 = response1.json()["predicted_price"]
        price2 = response2.json()["predicted_price"]
        
        assert price1 == price2, \
            "Lo stesso input deve produrre lo stesso output"


# =============================================================================
# Test per l'interfaccia Gradio (struttura base)
# =============================================================================

class TestGradioInterface:
    """Test per l'interfaccia Gradio."""
    
    def test_predict_price_function_exists(self):
        """Verifica che la funzione predict_price esista."""
        from app import predict_price
        assert callable(predict_price)
    
    def test_create_interface_function_exists(self):
        """Verifica che la funzione create_interface esista."""
        from app import create_interface
        assert callable(create_interface)
    
    def test_api_url_defined(self):
        """Verifica che API_URL sia definito."""
        from app import API_URL
        assert API_URL is not None
        assert "127.0.0.1" in API_URL or "localhost" in API_URL


# =============================================================================
# Test di integrazione
# =============================================================================

class TestIntegration:
    """Test di integrazione per il sistema completo."""
    
    @pytest.fixture(autouse=True)
    def setup_app(self, model_path):
        """Setup dell'app con il modello di test."""
        with patch('api.main.MODEL_PATH', model_path):
            if 'api.main' in sys.modules:
                del sys.modules['api.main']
            
            from api.main import app
            from fastapi.testclient import TestClient
            
            self.client = TestClient(app)
    
    def test_full_prediction_flow(self):
        """Test del flusso completo di predizione."""
        # Simula un utente che inserisce dati
        user_input = {
            "MedInc": 5.0,
            "HouseAge": 20.0,
            "AveRooms": 6.0,
            "AveBedrms": 1.2,
            "Population": 1000.0,
            "AveOccup": 2.5,
            "Latitude": 37.0,
            "Longitude": -121.5
        }
        
        # Invia richiesta
        response = self.client.post("/predict", json=user_input)
        
        # Verifica risposta
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))
        
        # Il prezzo dovrebbe essere ragionevole per California
        # (valori tipici sono tra 0.5 e 5.0, cioè $50k - $500k)
        assert 0 < data["predicted_price"] < 10, \
            f"Prezzo {data['predicted_price']} sembra fuori range per California"
    
    def test_edge_case_extreme_values(self):
        """Test con valori estremi."""
        extreme_input = {
            "MedInc": 15.0,  # Reddito molto alto
            "HouseAge": 1.0,  # Casa nuova
            "AveRooms": 10.0,  # Molte stanze
            "AveBedrms": 2.0,
            "Population": 100.0,  # Popolazione bassa (zona esclusiva)
            "AveOccup": 1.5,
            "Latitude": 37.8,  # San Francisco area
            "Longitude": -122.4
        }
        
        response = self.client.post("/predict", json=extreme_input)
        
        # Dovrebbe comunque funzionare
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
