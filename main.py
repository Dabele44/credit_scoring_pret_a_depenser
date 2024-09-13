from fastapi import FastAPI, HTTPException # pour créer l'API web et gérer les erreurs HTTP.
from pydantic import BaseModel # : pour valider les données d'entrée sous forme de modèles.
import joblib # pour charger le modèle
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb

# Chargement du modèle sérialisé
model = joblib.load('credit_scoring_new.joblib')

# Chargement du seuil optimal
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read())

# Initialisation de FastAPI
app = FastAPI() # Une instance de l'application FastAPI est créée 

# Modèle de données pour les prédictions
class InputData(BaseModel): # La classe InputData est définie pour valider les données d'entrée de l'API. 
    data: list[dict] # Elle attend une liste de dictionnaires qui seront convertis en DataFrame.


# Route pour vérifier si le serveur fonctionne
@app.get("/")
def home():
    return {"message": "API de scoring de crédit est en cours d'exécution."}

# Route pour prédire la classe d'un client
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # La route prend des données en entrée et les convertit en DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Elle prédit des probabilités
        y_pred_proba = model.predict_proba(df)[:, 1]
        
        # Elle applique le seuil optimal
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Et retourne la prédiction et la probabilité
        return {"prediction": y_pred.tolist(), "probability": y_pred_proba.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route pour obtenir l'explication SHAP des prédictions
@app.post("/explain")
def explain(input_data: InputData):
    try:
        # La route prend des données en entrée et les convertit en DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Elle initialise l'Explainer SHAP
        explainer = shap.Explainer(model.named_steps['model'])
        shap_values = explainer.shap_values(df)
        
        # Et retourne les valeurs SHAP pour la classe positive
        explanation = shap_values[1].tolist()  
        
        return {"shap_values": explanation, "base_value": base_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

