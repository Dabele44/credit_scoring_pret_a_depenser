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
test_data = pd.read_csv(data_path, index_col=False)
list_id_client=test_data.SK_ID_CURR.to_list()


# Chargement du dataset initial
initial_data_path = 'application_test.csv'
initial_test_data=pd.read_csv(initial_data_path)
initial_test_data_reduced = initial_test_data[initial_test_data['SK_ID_CURR'].isin(list_id_client)]
initial_test_data_reduced=initial_test_data_reduced[['SK_ID_CURR',
    'CODE_GENDER',
    'DAYS_BIRTH',
    'NAME_FAMILY_STATUS',
    'CNT_CHILDREN',
    'CNT_FAM_MEMBERS',
    'NAME_HOUSING_TYPE',
    'DAYS_EMPLOYED',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE']]
initial_test_data_reduced['AGE']=round(initial_test_data_reduced['DAYS_BIRTH']/-365,2)
initial_test_data_reduced['EMPLOYMENT_LENGTH']=round(initial_test_data_reduced['DAYS_EMPLOYED']/-365,2)
initial_test_data_reduced=initial_test_data_reduced.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1)

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
        font-size: 30px;
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
    .custom-title {
        font-size: 60px;  
        text-align: center;
        color: #333333;
        font-weight: bold;
        
    }
    .section-title {
        font-size: 40px;  
        color: #333333;
        font-weight: bold;
        text-decoration: underline;  
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ajout du titre "Credit Scoring" avec la classe personnalisée
st.markdown('<div class="custom-title"> CREDIT SCORING </div>', unsafe_allow_html=True)


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
    show_page1 = st.checkbox("Customer Information", value=True)
    show_page2 = st.checkbox("Customer Information Analysis")
    show_page3 = st.checkbox("Decision")

# Si l'ID est valide, exécuter le reste du code en fonction des cases cochées
if valid_id:

    # Affichage de l'ID du client sous le titre, en petit
    st.markdown(f'<p style="text-align:center; font-size:16px; color:#555;">Customer ID: {selected_id}</p>', unsafe_allow_html=True)

    selected_data = test_data[test_data['SK_ID_CURR'] == selected_id].iloc[:, 1:].copy()
    selected_data.reset_index(drop=True, inplace=True)

    # Filtrer les données du dataset initial pour l'ID client sélectionné
    selected_initial_data = initial_test_data_reduced[initial_test_data_reduced['SK_ID_CURR'] == selected_id]
    selected_initial_data.reset_index(drop=True, inplace=True)

    if show_page1:
        st.markdown('<div class="section-title">Customer Information :</div>', unsafe_allow_html=True)
        # Affichage des informations client
       
        if not selected_initial_data.empty:
            # Affichage des informations une par une
            customer_info = selected_initial_data.iloc[0]  # Sélectionne la première ligne (l'unique client)

            # Récupération du numéro de l'ID client
            customer_id = customer_info['SK_ID_CURR']

            # Affichage de la phrase avec le numéro d'ID du client
            st.write(f"Here are the details for this customer :")


            
            st.markdown(f"**Gender**: {customer_info['CODE_GENDER']}")
            st.markdown(f"**Family Status**: {customer_info['NAME_FAMILY_STATUS']}")
            st.markdown(f"**Number of Children**: {customer_info['CNT_CHILDREN']}")
            st.markdown(f"**Family Members**: {customer_info['CNT_FAM_MEMBERS']}")
            st.markdown(f"**Housing Type**: {customer_info['NAME_HOUSING_TYPE']}")
            st.markdown(f"**Age**: {customer_info['AGE']} years")
            st.markdown(f"**Employment Length**: {customer_info['EMPLOYMENT_LENGTH']} years")
            st.markdown(f"**Income**: $ {customer_info['AMT_INCOME_TOTAL']}")
            st.markdown(f"**Credit Amount**: $ {customer_info['AMT_CREDIT']}")
            st.markdown(f"**Annuity**: $ {customer_info['AMT_ANNUITY']}")
            st.markdown(f"**Goods Price**: $ {customer_info['AMT_GOODS_PRICE']}")
        else:
            st.write("No information available for the selected customer.")

    if show_page2:
        st.markdown('<div class="section-title">Customer Information Analysis :</div>', unsafe_allow_html=True)
        
        # Placeholder pour l'analyse des informations du client
        st.write("Analyse des informations du client ici...")

    if show_page3:
        st.markdown('<div class="section-title">Decision :</div>', unsafe_allow_html=True)

        # Le code existant pour les prédictions et l'analyse
        # Interroger l'API pour les prédictions
        api_response = get_predictions_from_api(selected_data.to_dict(orient='records'))

        if api_response is not None:
            prediction_proba = api_response['probability'][0]
            prediction = api_response['prediction'][0]

            # Affichage du texte coloré en fonction de la prédiction
            if prediction == 0:
                st.markdown("<div class='accepted-text'>- ACCEPTED -</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='rejected-text'>- REJECTED -</div>", unsafe_allow_html=True)

            # Séparation par une ligne
            st.markdown("---")

            # Ajout d'un titre centré pour "Prediction Probability :"
            st.markdown("<div class='centered-text'>Risk Probability :</div>", unsafe_allow_html=True)
            st.markdown("<div class='risk-text'>If the customer obtains a score < 0.91, we consider him to be risky (red line)</div>", unsafe_allow_html=True)

            # Affichage des résultats sous forme de jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=1-prediction_proba,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 0.91},
                gauge={
                    'axis': {'range': [0, 1]},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 1], 'color': "gray"}
                    ],  # Correct format for steps
                    'bar': {'color': "#1f77b4"},  # Couleur personnalisée pour la barre
                    'threshold': {
                        'line': {'color': "red", 'width': 10},
                        'thickness': 0.75,
                        'value': 0.91
                    }
                },
                title={'text': "", 'font': {'size': 14}}  # On ne met pas de titre ici car il est ajouté manuellement
            ))

            fig.update_layout(height=500)  # Réduire la hauteur de la jauge
            st.plotly_chart(fig, use_container_width=True)

            # Séparation par une ligne
            st.markdown("---")

            st.markdown("<div class='centered-text'>Features entering in the model :</div>", unsafe_allow_html=True)
            st.markdown("<div class='risk-text'>Here are the features for the selected customer (Use the horizontal scrollbar to see all the features) :</div>", unsafe_allow_html=True)

            st.dataframe(selected_data.reset_index(drop=True), use_container_width=True)


            # Séparation par une ligne
            st.markdown("---")

            # Obtenir les valeurs SHAP depuis l'API et les afficher
            shap_values = get_shap_values_from_api(selected_data.to_dict(orient='records'))
            if shap_values is not None:
                st.markdown("<div class='centered-text'>Feature Importance for this prediction :</div>", unsafe_allow_html=True)
                st.markdown("<div class='risk-text'>This section shows how each feature influenced the model's decision for the selected customer. Features that push the result to the right increase the likelihood of a high-risk prediction, while those that push it to the left reduce the risk. The further a factor moves in either direction, the more impact it has on the final decision. :</div>", unsafe_allow_html=True)

                display_shap_values(selected_data, shap_values)
        else:
            st.write("Error when retrieving predictions.")
