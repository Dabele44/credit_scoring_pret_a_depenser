from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb

# Charger le modèle sérialisé
model = joblib.load('credit_scoring_new.joblib')

# Charger le seuil optimal
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read())

# Initialiser FastAPI
app = FastAPI()

# Modèle de données pour les prédictions
class InputData(BaseModel):
    data: list[dict]

# Route pour vérifier si le serveur fonctionne
@app.get("/")
def home():
    return {"message": "API de scoring de crédit est en cours d'exécution."}

# Route pour prédire la classe d'un client
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convertir les données d'entrée en DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Prédiction des probabilités
        y_pred_proba = model.predict_proba(df)[:, 1]
        
        # Appliquer le seuil optimal
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Retourner la prédiction et la probabilité
        return {"prediction": y_pred.tolist(), "probability": y_pred_proba.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route pour obtenir l'explication SHAP des prédictions
@app.post("/explain")
def explain(input_data: InputData):
    try:
        # Convertir les données d'entrée en DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Initialisation de l'Explainer SHAP
        explainer = shap.Explainer(model.named_steps['model'])
        shap_values = explainer.shap_values(df)
        
        # Retourner les valeurs SHAP
        explanation = shap_values[1].tolist()  # Retourner les valeurs SHAP pour la classe positive
        
        return {"shap_values": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

