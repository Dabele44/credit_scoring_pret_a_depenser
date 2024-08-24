# Credit Scoring - Application de Prédiction de Risque de Crédit

**Lien vers le dashboard Streamlit** : [Credit Scoring Dashboard](https://dabele44-credit-scoring-pret-a-depe-app-streamlit-joblib-yq1wpc.streamlit.app/)

Ce projet consiste à créer une application de scoring de crédit pour la société Prêt à Dépenser. L'objectif est de prédire si un client est à risque à partir de ses données personnelles et financières, permettant ainsi à l'entreprise de décider d'accorder ou de refuser le crédit. 

## Table des matières

- [Contenu du projet](#contenu-du-projet)
- [Les données](#les-donnees)
- [Installation](#installation)
- [Modèle de Machine Learning](#modèle-de-machine-learning)
- [API Streamlit](#api-streamlit)
- [Tests Unitaires](#tests-unitaires)
- [Dépendances](#dépendances)
- [Structure du repository](#structure-du-repository)
- [Auteur](#auteur)

## Contenu du projet

Ce projet va au-delà du simple développement d'un modèle de machine learning. En effet, il intègre plusieurs aspects cruciaux du cycle de vie d'un projet de data science, en suivant une démarche **MLOps** (Machine Learning Operations) complète. Les attentes du projet incluent :

- **Préparation des données** : Transformation des features et sélection des variables les plus pertinentes.
- **Modélisation machine learning** : Utilisation d'un modèle de machine leearning, optimisé via GridSearch, pour obtenir des performances prédictives élevées tout en évitant le surapprentissage.
- **Vérification du Data Drift** : Mise en place d'une analyse de dérive des données (**Data Drift**) entre les jeux de données d'entraînement et de test à l'aide de la bibliothèque **Evidently**. Cette étape est essentielle pour garantir que le modèle reste performant sur de nouvelles données et pour identifier des changements dans les distributions des données.
- **Explications des prédictions** : Utilisation de la méthode **SHAP** pour fournir des explications détaillées des décisions du modèle, assurant ainsi la transparence et l'explicabilité des prédictions.
- **Tests unitaires** : Implémentation de tests unitaires pour valider chaque composant du pipeline, garantissant que le modèle fonctionne correctement avec des données nouvelles ou modifiées. Ces tests permettent de vérifier la structure des données, la cohérence des prédictions, et la justesse des valeurs calculées par SHAP.
- **Déploiement du modèle** : Création d'une interface interactive en utilisant, permettant à l'utilisateur final de soumettre des prédictions en temps réel et de visualiser les explications des décisions du modèle.

En somme, ce projet ne se limite pas à la création d'un modèle prédictif. Il s'inscrit dans une démarche rigoureuse d'industrialisation de la data science, en assurant à la fois la **robustesse** du modèle, la **transparence des prédictions**, et la **facilité de déploiement** et d'**interaction utilisateur**.

## Les données

Données Kaggle : https://www.kaggle.com/c/home-credit-default-risk/data

## Installation

### Prérequis

Assurez-vous que vous avez **Python 3.8 ou plus** installé sur votre machine.

### Étapes d'installation

1. Clonez ce dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/credit_scoring_pret_a_depenser.git
    cd credit_scoring_pret_a_depenser
    ```

2. Installez les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```
    
## Modèle de Machine Learning

Le modèle utilisé est un **LightGBM Classifier**, une implémentation très performante des méthodes de boosting basée sur les arbres de décision. Il a été choisi pour ce projet en raison de sa capacité à gérer efficacement de grandes quantités de données tout en maintenant de bonnes performances de prédiction. 

### Modèle final après optimisation

Le modèle est stocké dans un fichier `.joblib` et chargé dans l'application Streamlit pour effectuer des prédictions en temps réel.

## API Streamlit

1. Lancer l'API Streamlit en local :
    ```bash
    streamlit run app_streamlit_joblib.py
    ```

2. Une fois l'application lancée, un onglet de votre navigateur s'ouvrira avec l'application prête à être utilisée. 
   
### Fonctionnalités de l'application Streamlit

- **Sélection de l'ID du client** : Vous pouvez sélectionner un ID client dans le dataset ou entrer manuellement l'ID.
- **Prédiction du risque** : Le modèle affiche une jauge indiquant la probabilité de risque du client.
- **Explications SHAP** : Un graphique **waterfall** explique les contributions des principales caractéristiques à la décision de crédit.

L'interface est personnalisée avec un style CSS pour améliorer l'expérience utilisateur.


## Tests Unitaires

Les tests unitaires permettent de garantir la fiabilité du modèle de prédiction et de ses différentes composantes. Ils vérifient notamment que les données sont bien formatées, que les prédictions se situent dans des plages valides, et que les valeurs SHAP (expliquant les décisions du modèle) sont correctement calculées. Ces tests sont exécutés avant chaque déploiement afin de garantir que le modèle fonctionne comme attendu.

### Lancement des tests unitaires

1. Pour exécuter les tests unitaires, lancez la commande suivante :
    ```bash
    python -m unittest test_model.py
    ```

2. Les tests incluent :
    - Vérification du nombre de variables explicatives
    - Vérification des noms des variables
    - Vérification des formats des variables
    - Vérification que le threshold est bien de 0.09
    - Vérification du format des prédictions de probabilité
    - Vérification que les prédictions sont bien égales à 0 ou 1
    - Vérification du format de l'ID du client
    - Vérification de la longueur de l'ID du client
    - Vérification du nombre de valeurs SHAP
    - Vérification que la somme des valeurs SHAP plus le biais est proche de la prédiction à 5 décimales près


## Dépendances

Les principales dépendances pour ce projet sont listées dans le fichier `requirements.txt` :

- `streamlit==1.36.0`
- `pandas==2.1.4`
- `numpy==1.26.4`
- `shap==0.44.0`
- `mlflow==2.12.2`
- `matplotlib==3.8.0`
- `plotly==5.22.0`
- `imblearn==0.0`
- `lightgbm==4.3.0`

Installez-les en utilisant la commande :
```bash
pip install -r requirements.txt

## Structure du repository

Ce repository contient plusieurs fichiers et dossiers organisés comme suit :

```bash
├── .github/workflows/              # Dossier pour les workflows GitHub Actions (CI/CD)
├── .gitattributes                  # Fichier pour gérer les attributs spécifiques à Git (fin de ligne, etc.)
├── .gitignore                      # Fichier qui spécifie les fichiers à ignorer par Git
├── 00_Exploration_et_1ères_modélisations.ipynb    # Notebook pour l'exploration des données et les premières modélisations
├── 01_Feature_Engineering.ipynb     # Notebook pour le Feature Engineering (création et transformation des variables)
├── 02_Feature_Selection.ipynb       # Notebook pour la sélection des variables pertinentes
├── 03_Nouvelle_modelisation.ipynb   # Nouvelle phase de modélisation avec optimisation
├── 04_Evidently.ipynb               # Notebook pour la vérification du Data Drift avec la librairie Evidently
├── 05_Nouvelle_modelisation_sans_drift.ipynb  # Modélisation sans les variables ayant montré du Data Drift
├── README.md                        # Fichier de documentation du projet (ce fichier)
├── app_streamlit_joblib.py          # Script pour le déploiement de l'application avec Streamlit
├── bannière.png                     # Image utilisée dans l'interface utilisateur Streamlit
├── credit_scoring_new.joblib        # Modèle LightGBM sauvegardé après optimisation
├── data_drift_report_all.html       # Rapport Evidently sur le Data Drift (pour l'ensemble des variables)
├── data_drift_report_short.html     # Rapport Evidently sur le Data Drift (pour un sous-ensemble des variables)
├── optimal_threshold.txt            # Fichier contenant le seuil optimal de décision (threshold) utilisé par le modèle
├── reconstituted_test_sampled.csv   # Jeu de données de test échantillonné utilisé pour les tests de prédiction
├── requirements.txt                 # Fichier contenant la liste des dépendances du projet
├── schéma_tables.png                # Schéma illustrant les tables de données (si applicable)
├── to_merge.ipynb                   # Notebook pour grouper les notebooks en un seul notebook (il est demandé un livrable en un seul notebook)
├── unit_testing.py                  # Script contenant les tests unitaires pour valider le modèle


## Auteur
Ce projet a été réalisé par Anne BELOUARD. 

