import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import numpy as np 

# Chargement des données de test reconstituées
data_path = 'reconstituted_test_sampled.csv'
test_data = pd.read_csv(data_path)

# Fonction pour interroger l'API pour les prédictions
def get_predictions_from_api(input_data, api_url="https://scorecredit-93521a3704b4.herokuapp.com/predict"):
    response = requests.post(api_url, json={"data": input_data})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
        return None

# Fonction pour obtenir les valeurs SHAP depuis l'API
def get_shap_values_from_api(input_data, api_url="https://scorecredit-93521a3704b4.herokuapp.com/explain"):
    response = requests.post(api_url, json={"data": input_data})
    if response.status_code == 200:
        return response.json()['shap_values']
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
        return None

# Fonction pour afficher les valeurs SHAP pour l'individu sélectionné
def display_shap_values(input_data, shap_values):
    # On suppose que shap_values est une liste contenant les valeurs SHAP pour chaque feature
    plt.figure(figsize=(25, 10))
    shap.waterfall_plot(shap.Explanation(values=np.array(shap_values[0]),
                                         base_values=0,  # Base value is not fournie par l'API, set to 0
                                         data=input_data.iloc[0],
                                         feature_names=input_data.columns.tolist()), show=False)
    st.pyplot(plt, clear_figure=True)

# Chargement de l'optimal_threshold depuis le fichier texte
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read().strip())

# Interface Streamlit
st.sidebar.image("bannière.png", use_column_width=True)  # Ajout de la bannière en haut de la sidebar

st.markdown('<h1 style="text-align: center; color: #333333;">Credit Scoring</h1>', unsafe_allow_html=True)


# Changement de la couleur de fond de la sidebar et la police en blanc
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
        color: black !important;  /* Ensure text input is black */
    }
    .stSelectbox div {
        color: black !important;  /* Ensure selectbox text is black */
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
    # Ajout du titre en majuscules
    st.markdown("<div class='sidebar-title'>CHOOSE CUSTOMER ID</div>", unsafe_allow_html=True)

    # Sélection de la méthode d'entrée
    input_method = st.radio("Choose input method:", ('Selectbox', 'Text Input'))

    # Sélection d'un ID pour l'analyse
    selected_id = None  # Initialisation de selected_id pour gérer l'état initial
    valid_id = False     # Variable de contrôle pour la validité de l'ID

    # Si l'utilisateur choisit "Selectbox"
    if input_method == 'Selectbox':
        selected_id = st.selectbox("Select a customer ID and press Enter", test_data['SK_ID_CURR'])
        valid_id = True  # Si c'est un selectbox, l'ID est forcément valide

    # Si l'utilisateur choisit "Text Input"
    elif input_method == 'Text Input':
        selected_id = st.text_input("Enter a customer ID and press Enter", value="")

        # Si l'utilisateur a entré quelque chose dans le champ Text Input
        if selected_id:
            if selected_id.isdigit():  # Vérifier si la saisie est un nombre
                selected_id = int(selected_id)
                if selected_id in test_data['SK_ID_CURR'].values:
                    valid_id = True  # L'ID est valide s'il est dans le dataset
                else:
                    st.markdown("<p style='color:red;'>Please enter a valid ID from the dataset.</p>", unsafe_allow_html=True)  # L'ID n'est pas dans le dataset
            else:
                st.markdown("<p style='color:red;'>Please enter a valid ID from the dataset.</p>", unsafe_allow_html=True)  # Saisie non numérique

    # Ajout de la ligne de séparation ici
    st.markdown("---")  # Séparation visuelle entre "CHOOSE CUSTOMER ID" et "MENU"

    # Menu de sélection des pages avec un titre et une phrase explicative
    st.markdown("<div class='sidebar-title'>MENU</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px;'>Tick the page you want to see</p>", unsafe_allow_html=True)

    # Cases à cocher pour les pages
    show_page1 = st.checkbox("Customer Information")
    show_page2 = st.checkbox("Customer Information Analysis")
    show_page3 = st.checkbox("Decision")

# Si l'ID est valide, exécuter le reste du code en fonction des cases cochées
if valid_id:
    selected_data = test_data[test_data['SK_ID_CURR'] == selected_id].iloc[:, 1:]

    if show_page1:
        st.write("### Customer Information")
        # Placeholder pour les informations du client
        st.write("Informations sur le client ici...")

    if show_page2:
        st.write("### Customer Information Analysis")
        # Placeholder pour l'analyse des informations du client
        st.write("Analyse des informations du client ici...")

    if show_page3:
        st.write("### Decision")

        # Le code existant pour les prédictions et l'analyse
        # Interroger l'API pour les prédictions
        api_response = get_predictions_from_api(selected_data.to_dict(orient='records'))

        if api_response is not None:
            prediction_proba = api_response['probability'][0]
            prediction = api_response['prediction'][0]

            # Affichage du texte coloré en fonction de la prédiction
            if prediction == 0:
                st.markdown("<div class='accepted-text'>ACCEPTED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='rejected-text'>REJECTED</div>", unsafe_allow_html=True)

            # Séparation par une ligne
            st.markdown("---")

            # Ajout d'un titre centré pour "Prediction Probability :"
            st.markdown("<div class='centered-text'>Risk Probability :</div>", unsafe_allow_html=True)
            st.markdown("<div class='risk-text'>If the customer obtains a score >= 0.09, we consider him to be risky (red line)</div>", unsafe_allow_html=True)

            # Affichage des résultats sous forme de jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba,
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

            # Séparation par une ligne
            st.markdown("---")

            # Obtenir les valeurs SHAP depuis l'API et les afficher
            shap_values = get_shap_values_from_api(selected_data.to_dict(orient='records'))
            if shap_values is not None:
                st.markdown("<div class='centered-text'>Feature Importance for this prediction :</div>", unsafe_allow_html=True)
                display_shap_values(selected_data, shap_values)
        else:
            st.write("Error when retrieving predictions.")
