# Credit Scoring - Application de Prédiction de Risque de Crédit

**Lien vers le dashboard Streamlit** : [Credit Scoring Dashboard](https://dabele44-credit-scoring-pret-a-depe-app-streamlit-joblib-yq1wpc.streamlit.app/)

Ce projet consiste à créer une application de scoring de crédit pour la société Prêt à Dépenser. L'objectif est de prédire la probabilité qu'un client soit à risque pour accorder ou refuser un crédit. Cette application a été déployée en utilisant **Streamlit**, avec un modèle de machine learning de type **LightGBM**, et est accompagnée de tests unitaires pour garantir la qualité du modèle.

## Table des matières

- [Contexte du projet](#contexte-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API Streamlit](#api-streamlit)
- [Modèle de Machine Learning](#modèle-de-machine-learning)
- [Tests Unitaires](#tests-unitaires)
- [Dépendances](#dépendances)
- [Auteurs](#auteurs)

## Contexte du projet

L'objectif est de prédire si un client est à risque à partir de ses données personnelles et financières, permettant ainsi à l'entreprise de décider d'accorder ou de refuser le crédit. Le modèle utilisé dans ce projet est un **LightGBM** optimisé, et la visualisation des résultats est gérée via **Streamlit**. Le projet est conçu pour être interactif avec des explications **SHAP** des décisions du modèle.

### Fonctionnalités principales :

- **Prédiction du risque client** : Le modèle calcule la probabilité que le client soit à risque.
- **Explications SHAP** : Permet de visualiser les contributions des différentes caractéristiques à la prédiction.
- **Tests unitaires** : Validation des formats de données, des prédictions, des valeurs SHAP et de la qualité du modèle.

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

## Utilisation

1. Lancer l'API Streamlit en local :
    ```bash
    streamlit run app_streamlit_joblib.py
    ```

2. Une fois l'application lancée, un onglet de votre navigateur s'ouvrira avec l'application prête à être utilisée. Vous pouvez sélectionner un identifiant client pour prédire s'il est à risque ou non. L'application vous fournit également des explications détaillées sur la prédiction grâce aux valeurs **SHAP**.

### Fonctionnalités de l'application Streamlit

- **Sélection de l'ID du client** : Vous pouvez sélectionner un ID client dans le dataset ou entrer manuellement l'ID.
- **Prédiction du risque** : Le modèle affiche une jauge indiquant la probabilité de risque du client.
- **Explications SHAP** : Un graphique **waterfall** explique les contributions des principales caractéristiques à la décision de crédit.

L'interface est personnalisée avec un style CSS pour améliorer l'expérience utilisateur.

## API Streamlit

Le fichier `app_streamlit_joblib.py` contient tout le script Streamlit pour l'interface utilisateur. Voici un résumé des principales fonctionnalités incluses dans l'application :

- **Chargement du modèle** via `joblib`.
- **Génération de prédictions** avec la fonction `generate_predictions`.
- **Affichage des explications SHAP** pour chaque prédiction via `shap.waterfall_plot`.

L'interface vous permet d'interagir avec les prédictions du modèle de manière intuitive, avec un affichage clair des risques de crédit et des explications associées.

## Modèle de Machine Learning

Le modèle utilisé est un **LightGBM Classifier**, une implémentation très performante des méthodes de boosting basée sur les arbres de décision. Il a été choisi pour ce projet en raison de sa capacité à gérer efficacement de grandes quantités de données tout en maintenant de bonnes performances de prédiction. 

### Modèle final après optimisation

Le modèle est stocké dans un fichier `.joblib` et chargé dans l'application Streamlit pour effectuer des prédictions en temps réel.

## Tests Unitaires

Les tests unitaires permettent de garantir la fiabilité du modèle de prédiction et de ses différentes composantes. Ils vérifient notamment que les données sont bien formatées, que les prédictions se situent dans des plages valides, et que les valeurs SHAP (expliquant les décisions du modèle) sont correctement calculées. Ces tests sont exécutés avant chaque déploiement afin de garantir que le modèle fonctionne comme attendu.

### Lancement des tests unitaires

1. Pour exécuter les tests unitaires, lancez la commande suivante :
    ```bash
    python -m unittest test_model.py
    ```

2. Les tests incluent :
    - Vérification des formats des données et des colonnes.
    - Validation des prédictions (valeurs de sortie entre 0 et 1).
    - Vérification des valeurs **SHAP**.
    - Test du seuil optimal de 0.09.

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


## Auteurs
Ce projet a été réalisé par Anne BELOUARD. Vous pouvez me contacter via belouard@hotmail.com pour toute question.

