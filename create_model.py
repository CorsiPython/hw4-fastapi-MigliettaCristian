"""
Script per creare il modello pre-addestrato per l'Homework 4.

Questo script:
1. Carica il dataset California Housing
2. Addestra un modello di regressione Ridge con scaling
3. Salva il modello in formato pickle

Esegui questo script una volta per creare il file model/house_price_model.pkl
"""

import pickle
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def create_and_save_model():
    """Crea e salva il modello di predizione."""
    print("Caricamento dataset California Housing...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Dataset caricato: {X.shape[0]} campioni, {X.shape[1]} features")
    print(f"Features: {housing.feature_names}")
    
    # Crea una pipeline con scaling e regressione Ridge
    # TODO: Sperimenta con altri modelli o parametri se vuoi migliorare le prestazioni
    print("\nAddestramento modello...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Valutazione con cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"R² cross-validation: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Addestra sul dataset completo
    model.fit(X, y)
    
    # Salva il modello
    model_dir = Path(__file__).parent / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "house_price_model.pkl"
    
    print(f"\nSalvataggio modello in {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Verifica il modello salvato
    print("\nVerifica modello salvato...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Test predizione
    sample = X[0:1]
    prediction = loaded_model.predict(sample)[0]
    actual = y[0]
    
    print(f"Test predizione:")
    print(f"  Input: {dict(zip(housing.feature_names, sample[0]))}")
    print(f"  Predizione: ${prediction * 100000:.0f}")
    print(f"  Valore reale: ${actual * 100000:.0f}")
    
    print(f"\n✅ Modello salvato correttamente in: {model_path}")


if __name__ == "__main__":
    create_and_save_model()
