"""
Configurazione pytest per i test dell'Homework 4.

Questo file definisce fixtures comuni utilizzate dai test.
"""

import pytest
import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Crea un modello di test e restituisce il percorso.
    
    Questo modello viene creato una volta per sessione di test.
    
    Returns
    -------
    Path
        Percorso al file del modello di test
    """
    # Crea un modello semplice per i test
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Pipeline con scaling e regressione Ridge
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    model.fit(X, y)
    
    # Salva il modello
    model_dir = tmp_path_factory.mktemp("model")
    model_file = model_dir / "house_price_model.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    return model_file


@pytest.fixture(scope="session")
def trained_model(model_path):
    """Carica e restituisce il modello addestrato.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Il modello addestrato
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture
def sample_features():
    """Restituisce un dizionario di features di esempio.
    
    Returns
    -------
    dict
        Dizionario con le features per una predizione
    """
    return {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.0
    }


@pytest.fixture
def sample_features_list(sample_features):
    """Restituisce le features come lista ordinata.
    
    Returns
    -------
    list
        Lista delle features nell'ordine atteso dal modello
    """
    return [
        sample_features["MedInc"],
        sample_features["HouseAge"],
        sample_features["AveRooms"],
        sample_features["AveBedrms"],
        sample_features["Population"],
        sample_features["AveOccup"],
        sample_features["Latitude"],
        sample_features["Longitude"],
    ]
