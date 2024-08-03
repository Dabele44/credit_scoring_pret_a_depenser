#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import pandas as pd
import joblib
import shap
import numpy as np
from scipy.special import expit

# Rappel sur les notions de classe et d'instance :  
# class Voiture:  
#     def __init__(self, marque, modèle):  
#         self.marque = marque # self fait référence à l'instance actuelle  
#         self.modèle = modèle # self fait référence à l'instance actuelle  
# 
#     def afficher_détails(self):  
#         # self fait référence à l'instance actuelle  
#         print(f"Marque: {self.marque}, Modèle: {self.modèle}")  
# 
# # Création d'une instance de la classe Voiture  
# voiture1 = Voiture("Toyota", "Corolla")  
# voiture1.afficher_détails()  # Utilisation de l'instance courante pour appeler la méthode  
# 
# Lorsqu'on définit des méthodes à l'intérieur d'une classe, on utilise self pour faire référence à l'instance courante de la classe, c'est-à-dire l'objet spécifique qui appelle la méthode.

# In[2]:


# Chargement du modèle 
model_pipeline = joblib.load("credit_scoring_new.joblib")


# In[3]:


# Chargement des données de test reconstituées
data_path = 'reconstituted_test_sampled.csv'
test_data = pd.read_csv(data_path)
test_data.head()


# In[5]:


# De quel type sont mes variables
test_data.info()


# In[ ]:


# de quel type est ma variable ID
test_data.SK_ID_CURR.dtype


# In[9]:


# quelle est la longueur des ID
test_data['id_length'] = test_data['SK_ID_CURR'].apply(lambda x: len(str(x)))
test_data['id_length'].unique()


# In[ ]:


# Chargement de la fonction de prédiction
def generate_predictions(input_data, threshold):
    prediction_proba = model_pipeline.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction_proba, prediction


# Création d'une nouvelle classe appelée TestCreditScoringModel qui hérite de unittest.TestCase, ce qui signifie que notre classe TestCreditScoringModel peut utiliser toutes les méthodes fournies par  unittest.TestCase (assertEqual, assertTrue, assertIn, setUp, etc.).
# 
# Cette classe va contenir les tests unitaires que nous allons faire pour notre modèle de scoring de crédit.

# In[ ]:


class TestCreditScoringModel(unittest.TestCase):

    def setUp(self): # La méthode setUp est une méthode spéciale dans unittest qui est exécutée avant chaque méthode de test 
                        # pour préparer l'environnement de test. Elle est utilisée pour initialiser les données 
                        # et les paramètres nécessaires pour les tests
        
        # Lecture des données de test
        self.test_data = pd.read_csv(data_path) # En utilisant self, l'attribut self.test_data est accessible par toutes 
                                                # les méthodes de l'instance de la classe.
        
        # Chargement de l'optimal_threshold depuis le fichier texte
        with open('optimal_threshold.txt', 'r') as f:
            self.optimal_threshold = float(f.read().strip()) # Même chose : on ajoute self.devant optimal_threshold pour 
                                                            # rendre cet attribut accessible dans d'autres méthodes 
                                                            # de l'instance de la classe.

        # Chargement des données de test reconstituées
        self.test_data = pd.read_csv(data_path) # Même chose

        # Sélection d'un individu pour les tests
        self.selected_id = self.test_data['SK_ID_CURR'].iloc[0]
        self.selected_data = self.test_data[self.test_data['SK_ID_CURR'] == self.selected_id].iloc[:, 1:]

        # Initialisation de l'explainer SHAP
        self.explainer = shap.Explainer(model_pipeline.named_steps['model'])
        self.shap_values = self.explainer.shap_values(self.selected_data)
        

    ######################################################################################################################################################
    # Vérification du nombre de variables explicatives
    ######################################################################################################################################################
   
    def test_input_columns(self): # Cette ligne définit une méthode appelée test_input_columns 
                                    # au sein de la classe TestCreditScoringModel. 
                                    # self fait référence à l'instance actuelle de la classe.
        expected_num_columns = model_pipeline.named_steps['model'].n_features_
        actual_num_columns = self.selected_data.shape[1]
        self.assertEqual(actual_num_columns, expected_num_columns, f"Expected {expected_num_columns} columns, but got {actual_num_columns} columns.")
        # self.assertEqual(a, b) est une méthode d'assertion fournie par unittest.TestCase qui vérifie que a est égal à b.
        # Si les deux valeurs ne sont pas égales, le test échoue et le message spécifié est affiché : 
        # "Expected {expected_num_columns} columns, but got {actual_num_columns} columns.".


    
    
    ######################################################################################################################################################
    # Vérification des noms des variables
    ######################################################################################################################################################
   
    def test_variable_names(self):
        expected_columns = model_pipeline.named_steps['model'].feature_name_
        actual_columns = self.selected_data.columns.tolist()
        actual_columns = [col.replace(' ', '_') for col in actual_columns]
        self.assertListEqual(actual_columns, expected_columns, "The column names do not match the expected names.")



    
    ######################################################################################################################################################
    # Vérification des formats des variables
    ######################################################################################################################################################
    
    def test_variable_formats(self):
        for column in self.selected_data.columns:
            if self.selected_data[column].dtype not in [np.int64, np.float64]:
                self.fail(f"Column {column} has an unexpected data type: {self.selected_data[column].dtype}")



    
    ######################################################################################################################################################
    # Vérification que le threshold est bien de 0.09
    ######################################################################################################################################################

    def test_threshold(self):
        self.assertEqual(self.optimal_threshold, 0.09, "Optimal threshold is not equal to 0.09")



    
    ######################################################################################################################################################
    # Vérification du format des prédictions de probabilité
    ######################################################################################################################################################

    def test_prediction_proba_format(self):
        prediction_proba, _ = generate_predictions(self.selected_data, self.optimal_threshold)
        self.assertIsInstance(prediction_proba[0], np.float64, "Prediction probability is not a float")



    
    ######################################################################################################################################################
    # Vérification que les prédictions sont bien égales à 0 ou 1
    ######################################################################################################################################################

    def test_prediction_values(self):
        _, prediction = generate_predictions(self.selected_data, self.optimal_threshold)
        self.assertIn(prediction[0], [0, 1], "Prediction is not 0 or 1")



    
    ######################################################################################################################################################
    # Vérification du format de l'ID du client
    ######################################################################################################################################################

    def test_client_id_format(self):
        self.assertIsInstance(self.selected_id, np.int64, "Client ID is not an integer")



    
    ######################################################################################################################################################
    # Vérification de la longueur de l'ID du client
    ######################################################################################################################################################

    def test_client_id_lenght(self):
        self.assertEqual(len(str(self.selected_id)), 6, "Client ID must be 6 digits long")



    
    ######################################################################################################################################################
    # Vérification du nombre de valeurs SHAP
    ######################################################################################################################################################

    def test_shap_values_count(self):
        expected_shap_values_count = self.selected_data.shape[1]
        actual_shap_values_count = len(self.shap_values[1][0])
        self.assertEqual(actual_shap_values_count, expected_shap_values_count, f"Expected {expected_shap_values_count} SHAP values, but got {actual_shap_values_count}")



    
    ######################################################################################################################################################
    # Vérification que la somme des valeurs SHAP plus le biais est proche de la prédiction à 5 décimales près
    ######################################################################################################################################################

    def test_shap_values_sum(self):
        base_value = self.explainer.expected_value[1] # prédiction moyenne du modèle pour la classe positive si aucune caractéristique n'est incluse
        shap_values_sum = np.sum(self.shap_values[1][0]) #self.shap_values[1] donne les valeurs SHAP pour la classe positive (1). 
                                                        # [0] sélectionne l'échantillon en cours de test. 
                                                        # np.sum(self.shap_values[1][0]) calcule la somme de toutes les valeurs SHAP pour cet échantillon.
        prediction_proba, _ = generate_predictions(self.selected_data, self.optimal_threshold)
        predicted_value = prediction_proba[0] # probabilité prédite pour la classe positive (1).
        self.assertAlmostEqual(expit(base_value + shap_values_sum), predicted_value, places=5, msg="Sum of SHAP values plus base value does not match the predicted value")




if __name__ == '__main__':
    unittest.main()

