import streamlit as st
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Chargement du modèle depuis MLflow
logged_model = 'runs:/43be00c851724756aa0f6408ec39c83d/lgbm_classifier_model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Extraire le modèle LightGBM spécifique
lgbm_model = loaded_model._model_impl.lgb_model


# fonction shap
def generate_shap_waterfall(model, X_input, feature_names):
 
    # Create an explainer
    explainer = shap.Explainer(model, X_input)
    shap_values = explainer(X_input)
   
    
    return st_shap(shap.plots.waterfall(shap.Explanation(values=shap_values[0].values,
                                          base_values=shap_values[0].base_values,
                                          data=X_input.iloc[0],
                                          feature_names=feature_names), height=300))




# Interface utilisateur Streamlit
st.title("Credit Scoring Prediction")

# Création du formulaire pour les entrées utilisateur
st.header("Enter the details:")
EXT_SOURCE_1 = st.number_input("EXT_SOURCE_1", min_value=0.0, step=0.1)
EXT_SOURCE_2  = st.number_input("EXT_SOURCE_2", min_value=0.0, step=0.1)
EXT_SOURCE_3   = st.number_input("EXT_SOURCE_3", min_value=0.0, step=0.1)
DAYS_EMPLOYED    = st.number_input("DAYS_EMPLOYED", min_value=-10000.0, step=0.1)
DAYS_BIRTH    = st.number_input("DAYS_BIRTH", min_value=0.0, step=0.1)
client_installments_AMT_PAYMENT_min_sum    = st.number_input("client_installments_AMT_PAYMENT_min_sum", min_value=0.0, step=0.1)
bureau_DAYS_CREDIT_max    = st.number_input("bureau_DAYS_CREDIT_max", min_value=-10000.0, step=0.1)
bureau_DAYS_CREDIT_ENDDATE_max    = st.number_input("bureau_DAYS_CREDIT_ENDDATE_max", min_value=0.0, step=0.1)
client_cash_CNT_INSTALMENT_FUTURE_mean_max    = st.number_input("client_cash_CNT_INSTALMENT_FUTURE_mean_max", min_value=0.0, step=0.1)
OWN_CAR_AGE    = st.number_input("OWN_CAR_AGE", min_value=0.0, step=0.1)
bureau_AMT_CREDIT_SUM_DEBT_mean    = st.number_input("bureau_AMT_CREDIT_SUM_DEBT_mean", min_value=0.0, step=0.1)
DAYS_ID_PUBLISH     = st.number_input("DAYS_ID_PUBLISH", min_value=0.0, step=0.1)

# Bouton de prédiction
if st.button("Predict"):
    # Créer un DataFrame pour les entrées
    input_data = pd.DataFrame({
        'EXT_SOURCE_1': [EXT_SOURCE_1],
        'EXT_SOURCE_2': [EXT_SOURCE_2],
        'EXT_SOURCE_3': [EXT_SOURCE_3],
        'DAYS_EMPLOYED': [DAYS_EMPLOYED],
        'DAYS_BIRTH': [DAYS_BIRTH],
        'client_installments_AMT_PAYMENT_min_sum': [client_installments_AMT_PAYMENT_min_sum],
        'bureau_DAYS_CREDIT_max': [bureau_DAYS_CREDIT_max],
        'bureau_DAYS_CREDIT_ENDDATE_max': [bureau_DAYS_CREDIT_ENDDATE_max],
        'client_cash_CNT_INSTALMENT_FUTURE_mean_max': [client_cash_CNT_INSTALMENT_FUTURE_mean_max],
        'OWN_CAR_AGE': [OWN_CAR_AGE],
        'bureau_AMT_CREDIT_SUM_DEBT_mean': [bureau_AMT_CREDIT_SUM_DEBT_mean],
        'DAYS_ID_PUBLISH': [DAYS_ID_PUBLISH]
    })

    # Convertir les colonnes en types appropriés
    input_data['DAYS_BIRTH'] = input_data['DAYS_BIRTH'].astype('int64')
    input_data['DAYS_ID_PUBLISH'] = input_data['DAYS_ID_PUBLISH'].astype('int64')


    # Effectuer la prédiction
    prediction = loaded_model.predict(input_data)
    
    # Générer et afficher le graphique SHAP
    generate_shap_waterfall(lgbm_model, input_data, input_data.columns)
    

    # Afficher les résultats
    st.subheader("Prediction")
    st.write(f"Predicted Class: {prediction[0]}")
