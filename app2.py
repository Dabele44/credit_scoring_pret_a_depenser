import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow.pyfunc
import shap

# Chargement du modèle depuis MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
logged_model = 'runs:/b89a2f5c93d24c78a92835e16a4e2b1f/lgbm_classifier_model'  
loaded_model = mlflow.pyfunc.load_model(logged_model)

def generate_shap_waterfall(model, X_input, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_input)
   
    shap.initjs()
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    return fig

# Interface utilisateur Streamlit
st.title("Credit Scoring Prediction")

# Création du formulaire pour les entrées utilisateur
st.header("Enter the details:")
EXT_SOURCE_1 = st.number_input("EXT_SOURCE_1", min_value=0.0, step=0.1)
EXT_SOURCE_2 = st.number_input("EXT_SOURCE_2", min_value=0.0, step=0.1)
EXT_SOURCE_3 = st.number_input("EXT_SOURCE_3", min_value=0.0, step=0.1)
DAYS_EMPLOYED = st.number_input("DAYS_EMPLOYED", min_value=-10000, step=0.1)
DAYS_BIRTH = st.number_input("DAYS_BIRTH", min_value=0.0, step=0.1)
client_installments_AMT_PAYMENT_min_sum = st.number_input("client_installments_AMT_PAYMENT_min_sum", min_value=0.0, step=0.1)
bureau_DAYS_CREDIT_max = st.number_input("bureau_DAYS_CREDIT_max", min_value=0.0, step=0.1)
bureau_DAYS_CREDIT_ENDDATE_max = st.number_input("bureau_DAYS_CREDIT_ENDDATE_max", min_value=0.0, step=0.1)
client_cash_CNT_INSTALMENT_FUTURE_mean_max = st.number_input("client_cash_CNT_INSTALMENT_FUTURE_mean_max", min_value=0.0, step=0.1)
OWN_CAR_AGE = st.number_input("OWN_CAR_AGE", min_value=0.0, step=0.1)
bureau_AMT_CREDIT_SUM_DEBT_mean = st.number_input("bureau_AMT_CREDIT_SUM_DEBT_mean", min_value=0.0, step=0.1)
DAYS_ID_PUBLISH = st.number_input("DAYS_ID_PUBLISH", min_value=0.0, step=0.1)

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

    # Effectuer la prédiction
    prediction = loaded_model.predict(input_data)

    # Générer et afficher le graphique SHAP
    fig = generate_shap_waterfall(loaded_model, input_data, input_data.columns)
    st.pyplot(fig)

    # Afficher la prédiction
    st.subheader("Prediction")
    st.write(f"Predicted Class: {prediction[0]}")
