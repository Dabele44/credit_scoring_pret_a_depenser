# Implémentez un modèle de scoring
Projet n°7 du parcours Data Scientist OpenClassrooms en partenariat avec Centrale Supelec

Ce projet consiste à créer une application de scoring de crédit qui utilise des modèles de machine learning pour prédire si un client est à risque en se basant sur divers facteurs financiers et personnels. Le projet inclut la création des modèles, la détection de dérive des données, le déploiement de l'API avec FastAPI, ainsi que le test de l'API à la fois localement et après son déploiement sur Heroku.

## Table des matières

1. [Aperçu du projet](#aperçu-du-projet)
2. [Les données](#les-donnees)
3. [Contenu du dépôt](#contenu-du-depot)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
    - [Exécuter l'API localement](#exécuter-lapi-localement)
    - [Déployer l'API sur Heroku](#déployer-lapi-sur-heroku)
    - [Tester l'API](#tester-lapi)
    - [Utiliser l'interface Streamlit](#utiliser-linterface-streamlit)
6. [Formation et évaluation du modèle](#formation-et-évaluation-du-modèle)
7. [Détection de dérive des données](#détection-de-dérive-des-données)
8. [Tests unitaires](#tests-unitaires)
9. [Prérequis](#prérequis)
10. [Remerciements](#remerciements)
11. [Auteur](#auteur)


## Aperçu du projet

Ce projet consiste à développer un modèle de machine learning pour le scoring de crédit, à le déployer sous forme d'API, et à fournir une interface utilisateur pour l'interaction. L'API est développée en utilisant FastAPI et déployée sur Heroku. Le modèle de machine learning retenu est LightGBM et évalué à l'aide de métriques telles que l'AUC et le rappel.

Les principaux composants du projet incluent :
- **Formation du modèle :** Construction et formation d'un modèle LightGBM pour prédire le risque de crédit.
- **Détection de dérive des données :** Surveillance des données pour détecter des changements dans la distribution des données qui pourraient affecter les performances du modèle.
- **Développement de l'API :** Création d'une API avec FastAPI pour servir le modèle.
- **Déploiement :** Déploiement de l'API sur Heroku en utilisant GitHub Actions pour l'intégration continue et le déploiement continu (CI/CD).
- **Tests et validation :** Tests de l'API localement et après déploiement pour assurer sa fiabilité.

## Les données

Données Kaggle : [Home Credit](https://www.kaggle.com/c/home-credit-default-risk/data)

## Contenu du dépôt

Voici un aperçu des fichiers et dossiers présents dans ce dépôt :

- **`.github/workflows/`** : Contient les fichiers de configuration pour GitHub Actions, utilisés pour l'intégration continue et le déploiement automatique sur Heroku.
- **`.gitignore`** : Fichier qui spécifie les fichiers et dossiers à ignorer par Git (comme `__pycache__`, fichiers temporaires, etc.).
- **`00_Exploration_et_1ères_modélisations.ipynb`** : Notebook Jupyter contenant l'exploration initiale des données et les premières modélisations.
- **`01_Feature_Engineering.ipynb`** : Notebook pour la création et la transformation des caractéristiques à partir des données brutes.
- **`02_Feature_Selection.ipynb`** : Notebook pour la sélection des caractéristiques les plus pertinentes pour le modèle.
- **`03_Nouvelle_modelisation.ipynb`** : Notebook pour une nouvelle tentative de modélisation basée sur les caractéristiques sélectionnées.
- **`04_Evidently.ipynb`** : Notebook utilisant l'outil Evidently AI pour la détection de la dérive des données.
- **`05_Nouvelle_modelisation_sans_drift.ipynb`** : Notebook pour la modélisation après suppression des caractéristiques affectées par la dérive des données.
- **`Procfile`** : Fichier utilisé par Heroku pour définir comment exécuter l'application.
- **`credit_scoring_new.joblib`** : Fichier de modèle entraîné, sauvegardé avec Joblib pour une utilisation ultérieure.
- **`data_drift_report_all.html`** : Rapport complet de dérive des données généré par Evidently AI.
- **`data_drift_report_short.html`** : Rapport de dérive des données sur une sélection de variables, également généré par Evidently AI.
- **`main.py`** : Script principal pour exécuter l'API FastAPI.
- **`optimal_threshold.txt`** : Fichier texte contenant le seuil optimal de décision pour le modèle de scoring de crédit.
- **`reconstituted_test_sampled.csv`** : Jeu de données de test échantillonné pour les prédictions et l'évaluation du modèle.
- **`requirements.txt`** : Liste des dépendances Python nécessaires à l'exécution du projet.
- **`schéma_tables.png`** : Schéma représentant la structure des tables de données utilisées dans le projet.
- **`script_streamlit.py`** : Script pour exécuter l'interface utilisateur basée sur Streamlit.
- **`test_api.ipynb`** : Notebook pour tester l'API en envoyant des requêtes et en examinant les réponses.
- **`to_merge.ipynb`** : Notebook pour fusionner tous mes notebooks en un seul (demandé pour l'évaluation).
- **`unit_testing.py`** : Script contenant des tests unitaires pour vérifier la fonctionnalité des différents composants du projet.


## Installation

Pour installer et exécuter ce projet localement, suivez les étapes ci-dessous :

1. **Cloner le dépôt :**
    ```bash
    git clone https://github.com/Dabele44/credit_scoring_pret_a_depenser
    cd votre-projet
    ```

2. **Créer un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate # sur Windows, utilisez `venv\Scripts\activate`
    ```

3. **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Exécuter l'API localement

Pour exécuter l'API FastAPI localement :

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 
```
L'API sera disponible à l'adresse [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### Déployer l'API sur Heroku

Le déploiement de l'API sur Heroku est automatisé via GitHub Actions. Chaque fois que vous poussez des modifications sur la branche `main`, l'API est automatiquement déployée sur Heroku.

Pour configurer le déploiement :

1. Créez un compte sur Heroku.
2. Ajoutez les secrets suivants à votre dépôt GitHub (dans les paramètres des Actions GitHub) :
    - `HEROKU_API_KEY`
    - `HEROKU_EMAIL`
    - `HEROKU_APP_NAME`

Le fichier `deploy.yml` est configuré pour déclencher le déploiement sur Heroku chaque fois qu'une modification est poussée sur la branche `main`.

### Tester l'API

Vous pouvez tester l'API localement à l'aide du notebook `test_api.ipynb`. Ce notebook envoie des requêtes à l'API pour vérifier les prédictions et les explications SHAP.

### Utiliser l'interface Streamlit

Pour interagir avec le modèle via une interface graphique, exécutez l'application Streamlit :

```bash
streamlit run script_streamlit_plus_v7.py
```
L'application Streamlit vous permet de visualiser les prédictions du modèle et les valeurs SHAP pour chaque client. Cette application est déployée dans le Streamlit Community Cloud à cette adresse : https://dabele44-credit-scoring-pret-a--script-streamlit-plus-v7-ronrkx.streamlit.app/


## Création et évaluation du modèle

Le modèle est créé à l'aide de LightGBM sur des données de crédit. Les principales étapes de formation incluent :

- **Préparation des données :** Chargement et nettoyage des données, y compris la gestion des valeurs manquantes et la normalisation des caractéristiques.
- **Sélection des caractéristiques :** Réduction du nombre de caractéristiques en utilisant, entre autres, l'importance des caractéristiques.
- **Détection et suppression des variables affectées par la dérive des données :** Surveillance de la dérive des données entre les jeux de données d'entraînement et de test.
- **Optimisation des hyperparamètres :** Utilisation de GridSearchCV pour trouver les meilleurs hyperparamètres du modèle LightGBM.
- **Évaluation du modèle :** Évaluation du modèle sur un jeu de données de validation avec des métriques telles que l'AUC et le rappel.

## Détection de dérive des données

Le projet utilise `Evidently AI` pour surveiller la dérive des données entre le jeu de données d'entraînement et le jeu de données de test. Un rapport de dérive est généré pour identifier les caractéristiques qui changent significativement entre ces jeux de données, permettant ainsi de prévenir une dégradation des performances du modèle. 

## Tests unitaires

Des tests unitaires sont fournis pour vérifier :

- **Conformité des données en entrée :** Vérification que les colonnes et les types de données correspondent aux attentes du modèle.
- **Bon fonctionnement du modèle :** Vérification des sorties du modèle pour s'assurer qu'elles sont correctes.
- **Intégrité des valeurs SHAP :** Vérification que les valeurs SHAP sont cohérentes avec les prédictions.
- **Respect du seuil de risque optimal :** Vérification que le seuil de risque est appliqué correctement.

Ces tests peuvent être exécutés à l'aide de `unittest`


## Prérequis

Les principales bibliothèques utilisées dans ce projet sont :

- `streamlit==1.36.0`
- `pandas==2.1.4`
- `numpy==1.26.4`
- `shap==0.44.0`
- `mlflow==2.12.2`
- `matplotlib==3.8.0`
- `plotly==5.22.0`
- `imblearn==0.0`
- `lightgbm==4.3.0`
- `fastapi`
- `uvicorn`
- `requests`
- `pydantic`

Vous pouvez installer toutes les dépendances nécessaires en utilisant le fichier `requirements.txt` fourni avec ce projet. Pour ce faire, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## Remerciements

Merci à OpenClassrooms pour l'inspiration et les bases de ce projet. Un grand merci également à toutes les ressources en ligne et communautés de développeurs qui ont contribué à la réalisation de ce projet.

## Auteur

Anne BELOUARD
