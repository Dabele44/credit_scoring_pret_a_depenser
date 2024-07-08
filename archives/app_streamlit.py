import streamlit as st
import pandas as pd
import numpy as np
import shap
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# run_id 
run_id = "5f3f18e7487f41a592b94d6a228c2fac"

# Chemin relatif vers le modèle dans les artefacts de l'exécution
artifact_path = "lgbm_classifier_model"

# Créer un client MLflow
client = MlflowClient()

# Récupérer l'URI du modèle
model_uri = f"runs:/{run_id}/{artifact_path}"

model = mlflow.lightgbm.load_model(model_uri)


# Charger les données de test reconstituées
data_path = 'reconstituted_test.csv'
test_data = pd.read_csv(data_path)

# Fonction pour générer des prédictions
def generate_predictions(input_data, threshold):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction_proba = np.round(prediction_proba,2) 
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction_proba, prediction

# Fonction pour afficher les explications SHAP
def display_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    individual_shap = explainer.shap_values(input_data)
    predicted_class=prediction[0]
    shap_values_for_class=individual_shap[predicted_class]
    shap.plots.waterfall(shap.Explanation(values=shap_values_for_class[0], 
                                          base_values=explainer.expected_value[predicted_class], 
                                          data=input_data.iloc[0], 
                                          feature_names=input_data.columns.tolist()))

# Récupérer l'optimal_threshold depuis MLflow
def get_optimal_threshold(run_id, metric_name):
    client = MlflowClient()
    metric = client.get_metric_history(run_id, metric_name)
    return metric[0].value
    

# Récupérer l'optimal_threshold
optimal_threshold = get_optimal_threshold(run_id, "optimal_threshold")

# Interface Streamlit
st.title('Credit Scoring Model Deployment')


# Sélection d'un ID pour l'analyse
selected_id = st.selectbox("Select ID for Prediction and Explanation", test_data['SK_ID_CURR'])

# Sélection des données de l'individu
selected_data = test_data[test_data['SK_ID_CURR'] == selected_id].iloc[:, 1:]

# Générer les prédictions pour l'individu sélectionné
prediction_proba, prediction = generate_predictions(selected_data, optimal_threshold)

# Afficher les résultats
st.write(f"Prediction Probability: {prediction_proba[0]}")
st.write(f"Prediction: {'Approved' if prediction[0] == 0 else 'Not Approved'}")

# Afficher les explications SHAP pour l'individu sélectionné
st.write("Feature Importance for this prediction:")
display_shap_values(model, selected_data)

