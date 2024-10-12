import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import numpy as np 

# Définition de la langue de la page en utilisant une balise HTML (utile si le contenu est multilingue ou pour certaines balises)
st.markdown("""
    <html lang="en">
    </html>
    """, unsafe_allow_html=True)

# Chargement des données de test reconstituées (dataset avec des données test échantillonnées)
data_path = 'reconstituted_test_sampled.csv'
test_data = pd.read_csv(data_path, index_col=False)
list_id_client = test_data.SK_ID_CURR.to_list()

# Chargement du dataset initial (données de test initiales échantillonnées)
initial_data_path = 'application_test_sampled.csv'
initial_test_data = pd.read_csv(initial_data_path)

# Réduction du dataset initial aux colonnes pertinentes pour l'affichage
initial_test_data_reduced = initial_test_data[['SK_ID_CURR',
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

# Transformation des jours en années pour l'âge et la durée d'emploi
initial_test_data_reduced['AGE'] = round(initial_test_data_reduced['DAYS_BIRTH'] / -365, 2)
initial_test_data_reduced['EMPLOYMENT_LENGTH'] = round(initial_test_data_reduced['DAYS_EMPLOYED'] / -365, 2)

# Suppression des colonnes non nécessaires après la transformation
initial_test_data_reduced = initial_test_data_reduced.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1)

# Chargement des importances globales des features (données sur l'importance moyenne des features globalement)
global_importance = pd.read_csv('global_feature_importance.csv')

# Fonction pour comparer les importances locales (SHAP values pour un client) et globales, limitée aux 10 variables les plus importantes
def compare_global_local(selected_data, shap_values_local):
    # Extraction des valeurs SHAP locales (seules les valeurs absolues sont utilisées)
    local_shap_values = np.abs(np.array(shap_values_local[0]))

    # Tri des importances globales (valeurs déjà absolues)
    global_importance_sorted = global_importance.set_index('Feature').reindex(selected_data.columns).reset_index()

    # Création d'un DataFrame pour la comparaison des importances locales et globales
    feature_importance_df = pd.DataFrame({
        'Feature': selected_data.columns,
        'Local Importance (Abs)': local_shap_values,  # Importance locale en valeur absolue
        'Global Importance (Abs)': global_importance_sorted['MeanAbsSHAP'].values  # Importance globale en valeur absolue
    })

    # Tri des 10 features les plus importantes par importance globale décroissante
    feature_importance_df.sort_values(by='Global Importance (Abs)', ascending=False, inplace=True)
    feature_importance_df = feature_importance_df.head(10)

    # Affichage de texte expliquant la comparaison avant le graphique
    st.markdown("<p style='color: #555;'>The following graph compares the magnitude of feature importance at two levels: global (for all clients) and local (for a specific client).</p>", unsafe_allow_html=True)

    # Affichage du graphique comparatif (barres décalées pour éviter la superposition)
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_width = 0.4
    y_pos = np.arange(len(feature_importance_df['Feature']))

    # Barres pour l'importance globale et locale
    ax.barh(y_pos, feature_importance_df['Global Importance (Abs)'], bar_width, color='#006d5b', alpha=0.6, label='Global Importance (Abs)')
    ax.barh(y_pos + bar_width, feature_importance_df['Local Importance (Abs)'], bar_width, color='#ff8c00', alpha=0.6, label='Local Importance (Abs)')

    # Ajout des labels et inversion de l'axe Y pour avoir la feature la plus importante en haut
    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(feature_importance_df['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Absolute)')
    ax.set_title('Comparison of Top 10 Local and Global Feature Importances (Absolute Values)')
    ax.legend()

    # Affichage du graphique
    st.pyplot(fig)

    # Explication après le graphique pour guider l'utilisateur dans l'interprétation
    st.markdown("""
    <p style='color: #555;'>
    Here’s how to interpret it:
    
    **Vertical axis (Features):**
    This axis lists the top 10 most important variables, ranked by their global importance in descending order (from top to bottom).
    
    **Horizontal axis (Importance):**
    This axis shows the magnitude of each feature's importance, comparing global and local levels (in absolute terms).

    **Green bars (Global Importance - Absolute):**
    The green bars represent the average absolute importance of the features in the model, calculated across all clients.
    
    **Yellow bars (Local Importance - Absolute):**
    The yellow bars show the absolute impact of the feature on the prediction for the selected client.
    
    </p>
    """, unsafe_allow_html=True)

# Fonction pour interroger l'API et obtenir les prédictions
def get_predictions_from_api(input_data, api_url="https://scorecredit-93521a3704b4.herokuapp.com/predict"):
    response = requests.post(api_url, json={"data": input_data})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
        return None

# Fonction pour obtenir les valeurs SHAP locales depuis l'API
def get_shap_values_from_api(input_data, api_url="https://scorecredit-93521a3704b4.herokuapp.com/explain"):
    response = requests.post(api_url, json={"data": input_data})
    if response.status_code == 200:
        return response.json()['shap_values']
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
        return None

# Fonction pour afficher les valeurs SHAP locales sous forme de graphique waterfall pour un client spécifique
def display_shap_values(input_data, shap_values):
    plt.figure(figsize=(25, 10))
    shap.waterfall_plot(shap.Explanation(values=np.array(shap_values[0]),
                                         base_values=0,  # Valeur de base par défaut
                                         data=input_data.iloc[0],
                                         feature_names=input_data.columns.tolist()), show=False)
    st.pyplot(plt, clear_figure=True)

# Chargement du seuil optimal pour la prédiction depuis un fichier texte
with open('optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read().strip())

# Interface Streamlit : affichage de la bannière dans la sidebar
st.sidebar.image("bannière.png", use_column_width=True, caption="Banner representing credit scoring application")

# Ajout du focus visuel pour améliorer l'accessibilité clavier
st.markdown(
    """
    <style>
    /* Ajoute un contour visible pour les éléments en focus */
    input:focus, select:focus, button:focus {
        outline: 2px solid #ff8c00;  /* Couleur visible lors du focus */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Changement de la couleur de fond de la sidebar et des éléments en blanc
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
        color: black !important;  /* Couleur du texte de l'input */
    }
    .stSelectbox div {
        color: black !important;  /* Couleur du texte de la selectbox */
    }
    .risk-text {
        color: #555;
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
    .bivariate-title {
        font-size: 32px;  
        color: #333333;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ajout du titre "Credit Scoring" en haut de la page avec la classe personnalisée
st.markdown('<h1 class="custom-title">CREDIT SCORING</h1>', unsafe_allow_html=True)

# Contenu de la sidebar
with st.sidebar:
    # Ajout du titre en majuscules
    st.markdown("<h2 class='sidebar-title'>CHOOSE CUSTOMER ID</h2>", unsafe_allow_html=True)

    # Sélection de la méthode d'entrée (via un selectbox ou un input textuel)
    input_method = st.radio("Choose input method:", ('Selectbox', 'Text Input'))

    # Initialisation des variables pour gérer l'ID client et sa validité
    selected_id = None  
    valid_id = False    

    # Si l'utilisateur choisit "Selectbox"
    if input_method == 'Selectbox':
        selected_id = st.selectbox("Select a customer ID and press Enter", test_data['SK_ID_CURR'])
        valid_id = True  # Si c'est un selectbox, l'ID est valide par défaut

    # Si l'utilisateur choisit "Text Input"
    elif input_method == 'Text Input':
        selected_id = st.text_input("Enter a customer ID and press Enter", value="")

        # Vérification si l'entrée est numérique et appartient au dataset
        if selected_id:
            if selected_id.isdigit():  
                selected_id = int(selected_id)
                if selected_id in test_data['SK_ID_CURR'].values:
                    valid_id = True  # L'ID est valide s'il est dans le dataset
                else:
                    st.markdown("<p style='color:red;'>Please enter a valid ID from the dataset.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:red;'>Please enter a valid ID from the dataset.</p>", unsafe_allow_html=True)

    # Ajout d'une ligne de séparation pour l'esthétique
    st.markdown("---")

    # Menu de sélection des pages
    st.markdown("<h3 class='sidebar-title'>MENU</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px;'>Tick the page you want to see</p>", unsafe_allow_html=True)

    # Cases à cocher pour les différentes pages
    show_page1 = st.checkbox("Customer Information", value=True)
    show_page2 = st.checkbox("Feature Visualization")
    show_page3 = st.checkbox("Decision")

# Si l'ID est valide, exécuter le reste du code en fonction des cases cochées
if valid_id:

    # Affichage de l'ID du client sous le titre, en petit
    st.markdown(f'<p style="text-align:center; font-size:16px; color:#555;">Customer ID: {selected_id}</p>', unsafe_allow_html=True)

    # Toujours récupérer les données sélectionnées indépendamment des pages cochées
    selected_data = test_data[test_data['SK_ID_CURR'] == selected_id].iloc[:, 1:].copy()
    selected_data.reset_index(drop=True, inplace=True)

    # Filtrer les données du dataset initial pour l'ID client sélectionné
    selected_initial_data = initial_test_data_reduced[initial_test_data_reduced['SK_ID_CURR'] == selected_id]
    selected_initial_data.reset_index(drop=True, inplace=True)

    # Page 1 : Afficher les informations du client si "Customer Information" est coché
    if show_page1:
        st.markdown('<h2 class="section-title">Customer Information :</h2>', unsafe_allow_html=True)
        # Affichage des informations client
       
        if not selected_initial_data.empty:
            customer_info = selected_initial_data.iloc[0]  # Sélectionne la première ligne (l'unique client)
            
            # Affichage des informations une par une
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

    # Page 2 : Afficher la visualisation des features si "Feature Visualization" est coché
    if show_page2:
        st.markdown('<h2 class="section-title">Feature Visualization :</h2>', unsafe_allow_html=True)

        # Exclusion de 'SK_ID_CURR' de la liste déroulante
        selectable_columns = [col for col in initial_test_data_reduced.columns if col != 'SK_ID_CURR']

        # Sélection de la première variable à visualiser
        selected_variable = st.selectbox(
            "Select a variable to visualize the distribution:",
            options=selectable_columns,
            index=0  
        )

        st.markdown(f"<p style='color: grey;'>The following graph shows the distribution of {selected_variable} for all customers and highlights the value for the selected customer.</p>", unsafe_allow_html=True)

        if not selected_initial_data.empty:
            # Vérification si la variable sélectionnée est numérique ou catégorielle
            if pd.api.types.is_numeric_dtype(initial_test_data_reduced[selected_variable]):
                # Graphique pour les variables numériques (histogramme)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(initial_test_data_reduced[selected_variable], bins=30, color='#ff8c00', alpha=0.7, label='All Customers')

                # Ajout de la position du client sélectionné
                client_value = selected_initial_data[selected_variable].values[0]
                ax.axvline(client_value, color='green', linestyle='dashed', linewidth=2, label=f'Selected Customer ({client_value})')

                ax.set_title(f'Distribution of {selected_variable}')
                ax.set_xlabel(selected_variable)
                ax.set_ylabel('Frequency')
                ax.legend()

                st.pyplot(fig)

            else:
                # Graphique pour les variables catégorielles (diagramme à barres)
                fig, ax = plt.subplots(figsize=(10, 6))

                # Comptage des occurrences de chaque catégorie
                value_counts = initial_test_data_reduced[selected_variable].value_counts()

                # Affichage du diagramme à barres pour toutes les catégories
                ax.bar(value_counts.index, value_counts.values, color='#ff8c00', alpha=0.7)

                ax.set_title(f'Distribution of {selected_variable}')
                ax.set_xlabel(selected_variable)
                ax.set_ylabel('Count')

                # Rotation des étiquettes de l'axe X
                plt.xticks(rotation=45)

                st.pyplot(fig)

                # Affichage de la valeur de la catégorie pour l'individu sélectionné
                client_value = selected_initial_data[selected_variable].values[0]
                st.markdown(f"<p style='color: #555;'>Selected Customer's {selected_variable}: <strong>{client_value}</strong></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:red;'>Please select a valid customer ID before proceeding with visualization.</p>", unsafe_allow_html=True)

        # Sélection des deux variables pour l'analyse bivariée
        st.markdown('<h3 class="bivariate-title">Bivariate Analysis:</h3>', unsafe_allow_html=True)

        variable1 = st.selectbox("Select the first variable for bivariate analysis", options=selectable_columns, index=0)
        variable2 = st.selectbox("Select the second variable for bivariate analysis", options=selectable_columns, index=1)

        st.markdown(f"<p style='color: grey;'>The following graph shows the relationship between {variable1} and {variable2} for all customers.</p>", unsafe_allow_html=True)

        # Vérification des types des variables sélectionnées
        if pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable1]) and pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable2]):
            # Graphique de dispersion pour deux variables numériques
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(initial_test_data_reduced[variable1], initial_test_data_reduced[variable2], color='#ff8c00', alpha=0.6, label='All Customers')

            # Ajout de la position du client sélectionné
            client_value1 = selected_initial_data[variable1].values[0]
            client_value2 = selected_initial_data[variable2].values[0]
            ax.scatter(client_value1, client_value2, color='green', s=100, label=f'Selected Customer ({client_value1}, {client_value2})')

            ax.set_title(f'{variable1} vs {variable2}')
            ax.set_xlabel(variable1)
            ax.set_ylabel(variable2)
            ax.legend()

            st.pyplot(fig)

        elif pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable1]) and not pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable2]):
            # Boxplot pour une variable numérique et une variable catégorielle
            fig, ax = plt.subplots(figsize=(10, 6))
            initial_test_data_reduced.boxplot(column=variable1, by=variable2, ax=ax, grid=False)
            ax.set_title(f'{variable1} distribution by {variable2}')
            ax.set_xlabel(variable2)
            ax.set_ylabel(variable1)
            st.pyplot(fig)

        elif pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable2]) and not pd.api.types.is_numeric_dtype(initial_test_data_reduced[variable1]):
            # Inverser les rôles des variables si la deuxième est numérique et la première catégorielle
            fig, ax = plt.subplots(figsize=(10, 6))
            initial_test_data_reduced.boxplot(column=variable2, by=variable1, ax=ax, grid=False)
            ax.set_title(f'{variable2} distribution by {variable1}')
            ax.set_xlabel(variable1)
            ax.set_ylabel(variable2)
            st.pyplot(fig)

        else:
            st.write("Bivariate analysis is only available for at least one numerical variable.")

    # Page 3 : Afficher la décision si "Decision" est coché
    if show_page3:
        st.markdown('<h2 class="section-title">Decision :</h2>', unsafe_allow_html=True)

        if not selected_data.empty:
            # Interroger l'API pour obtenir la prédiction et la probabilité
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

                # Affichage de la probabilité de risque avec un titre centré
                st.markdown("<h3 class='centered-text'>Risk Probability :</h3>", unsafe_allow_html=True)
                st.markdown("<p class='risk-text'>If the customer obtains a score < 0.91, we consider him to be risky (red line)</p>", unsafe_allow_html=True)

                # Affichage des résultats sous forme de jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=1-prediction_proba,
                    delta={'reference': 0.91},
                    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#1f77b4"}, 'threshold': {'line': {'color': "red", 'width': 10}, 'value': 0.91}},
                    title={'text': "", 'font': {'size': 14}}))
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Séparation par une ligne
                st.markdown("---")

                # Affichage des features utilisées dans le modèle
                st.markdown("<h3 class='centered-text'>Features entering in the model :</h3>", unsafe_allow_html=True)
                st.dataframe(selected_data.reset_index(drop=True), use_container_width=True)

                # Séparation par une ligne
                st.markdown("---")

                # Obtenir et afficher les valeurs SHAP depuis l'API
                shap_values = get_shap_values_from_api(selected_data.to_dict(orient='records'))
                if shap_values is not None:
                    st.markdown("<h3 class='centered-text'>Feature Importance for this prediction :</h3>", unsafe_allow_html=True)
                    display_shap_values(selected_data, shap_values)

                    # Comparaison des importances locales et globales
                    compare_global_local(selected_data, shap_values)
                else:
                    st.write("Error when retrieving SHAP values.")
            else:
                st.write("Error when retrieving predictions.")
        else:
            st.markdown("<p style='color:red;'>Please select a valid customer ID before proceeding with the decision analysis.</p>", unsafe_allow_html=True)
