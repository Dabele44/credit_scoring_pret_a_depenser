import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import imblearn




# Chargement du modèle
model_pipeline = joblib.load("credit_scoring.joblib")

# Extraire le modèle LGBM de la pipeline
model = model_pipeline.named_steps['model']

# Charger les données de test reconstituées
data_path = 'reconstituted_test.csv'
test_data = pd.read_csv(data_path)

# Fonction pour générer des prédictions
def generate_predictions(input_data, threshold):
    prediction_proba = model_pipeline.predict_proba(input_data)[:, 1]
    prediction_proba = np.round(prediction_proba, 2)
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction_proba, prediction

# Fonction pour afficher les explications SHAP
def display_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    individual_shap = explainer.shap_values(input_data)
    predicted_class = prediction[0]
    shap_values_for_class = individual_shap[predicted_class]
    shap.initjs()
    plt.figure(figsize=(25, 10))  # Augmenter la taille de la figure
    shap.waterfall_plot(shap.Explanation(values=shap_values_for_class[0], 
                                          base_values=explainer.expected_value[predicted_class], 
                                          data=input_data.iloc[0], 
                                          feature_names=input_data.columns.tolist()), show=False)
    st.pyplot(plt, clear_figure=True)  # Ajuster la taille de l'affichage

def st_shap(plot, height=400):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Charger l'optimal_threshold depuis le fichier texte
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read().strip())

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