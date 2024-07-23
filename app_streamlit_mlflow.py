import streamlit as st
import pandas as pd
import numpy as np
import shap
import mlflow
from mlflow.tracking import MlflowClient
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go



mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


# run_id
run_id = "5f3f18e7487f41a592b94d6a228c2fac"

# Chemin relatif vers le modèle dans les artefacts de l'exécution
artifact_path = "lgbm_classifier_model"

# Créer un client MLflow
client = MlflowClient()

# Récupérer l'URI du modèle
model_uri = f"runs:/{run_id}/{artifact_path}"

# Chargement du modèle
model = mlflow.lightgbm.load_model(model_uri)

# Charger les données de test reconstituées
data_path = 'reconstituted_test_sampled.csv'
test_data = pd.read_csv(data_path)

# Fonction pour générer des prédictions
def generate_predictions(input_data, threshold):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction_proba = np.round(prediction_proba, 2)
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction_proba, prediction

# Fonction pour afficher les explications SHAP
def display_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    individual_shap = explainer.shap_values(input_data)
    shap_values_for_class_1 = individual_shap[1]
    # shap.initjs()
    plt.figure(figsize=(25, 10))  # Augmenter la taille de la figure
    shap.waterfall_plot(shap.Explanation(values=shap_values_for_class_1[0], 
                                          base_values=explainer.expected_value[1], 
                                          data=input_data.iloc[0], 
                                          feature_names=input_data.columns.tolist()), show=False)
    st.pyplot(plt, clear_figure=True)  # Ajuster la taille de l'affichage

def st_shap(plot, height=400):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Fonction pour récupérer l'optimal_threshold depuis MLflow
def get_optimal_threshold(run_id, metric_name):
    client = MlflowClient()
    metric = client.get_metric_history(run_id, metric_name)
    return metric[0].value

# Récupérer l'optimal_threshold
optimal_threshold = get_optimal_threshold(run_id, "optimal_threshold")

# Interface Streamlit
st.sidebar.image("bannière.png", use_column_width=True)  # Ajouter la bannière en haut de la sidebar

st.markdown('<h1 style="text-align: center; color: #333333;">Credit Scoring</h1>', unsafe_allow_html=True)

# Ajouter un espacement avant le contenu de la sidebar
for _ in range(8):  # Ajouter plusieurs lignes vides pour augmenter l'espacement
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Changer la couleur de fond de la sidebar et la police en blanc
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #2c7353;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    .accepted-text {
        color: green;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .rejected-text {
        color: red;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .sidebar-title {
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 20px;
    }
    .centered-text {
        color: #0b141a;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stTextInput input {
        color: black !important;
    }
    .risk-text {
        color: grey;
        font-size: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Contenu de la sidebar
with st.sidebar:
    # Ajouter le titre en majuscules
    st.markdown("<div class='sidebar-title'>CHOOSE CUSTOMER ID</div>", unsafe_allow_html=True)
    
    # Sélection de la méthode d'entrée
    input_method = st.radio("Choose input method:", ('Selectbox', 'Text Input'))

    # Sélection d'un ID pour l'analyse
    if input_method == 'Selectbox':
        st.markdown("### **<span style='font-size:20px'>Select Customer Id </span>**", unsafe_allow_html=True)
        selected_id = st.selectbox("", test_data['SK_ID_CURR'])
    else:
        st.markdown("### **<span style='font-size:20px'>Enter Customer Id</span>**", unsafe_allow_html=True)
        selected_id = st.text_input("", value="")

# Vérifiez si l'ID entré est valide
if input_method == 'Selectbox' or (input_method == 'Text Input' and selected_id.isdigit()):
    selected_id = int(selected_id)
    if selected_id in test_data['SK_ID_CURR'].values:
        selected_data = test_data[test_data['SK_ID_CURR'] == selected_id].iloc[:, 1:]

        # Générer les prédictions pour l'individu sélectionné
        prediction_proba, prediction = generate_predictions(selected_data, optimal_threshold)

        # Afficher le texte coloré en fonction de la prédiction
        if prediction[0] == 0:
            st.markdown("<div class='accepted-text'>ACCEPTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rejected-text'>REJECTED</div>", unsafe_allow_html=True)

        # Séparer par une ligne
        st.markdown("---")

        # Ajouter un titre centré pour "Prediction Probability :"
        st.markdown("<div class='centered-text'>Prediction Probability :</div>", unsafe_allow_html=True)
        
        # Afficher les résultats sous forme de jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#1f77b4"},  # Couleur personnalisée pour la barre
                'threshold': {
                    'line': {'color': "red", 'width': 1},
                    'thickness': 1,
                    'value': 0.09
                }
            },
            title={'text': "", 'font': {'size': 14}}  # On ne met pas de titre ici car il est ajouté manuellement
        ))

        fig.update_layout(height=400)  # Réduire la hauteur de la jauge
        st.plotly_chart(fig, use_container_width=True)

        # Ajouter le texte centré en gris sous la jauge
        st.markdown("<div class='risk-text'>If the customer obtains a score greater than, or equal, to 0.09, we consider him to be risky</div>", unsafe_allow_html=True)

        # Séparer par une ligne
        st.markdown("---")

        # Afficher les explications SHAP pour l'individu sélectionné
        st.markdown("<div class='centered-text'>Feature Importance for this prediction :</div>", unsafe_allow_html=True)
        display_shap_values(model, selected_data)
    else:
        st.write("Please enter a valid ID from the dataset.")
else:
    st.write("Please enter a valid ID.")
