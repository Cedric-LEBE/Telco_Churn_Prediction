# Prediction du Churn Client avec PySpark (Databricks)

Projet de machine learning pour predire le **churn** (désabonnement) de clients telecom avec **PySpark** sur **Databricks**.

## Objectif

Construire un pipeline ML complet pour identifier les clients les plus susceptibles de resilier leur abonnement, afin de permettre des actions de retention ciblées.

## Contexte metier

Une entreprise telecom souhaite anticiper les départs clients (churn) pour reduire la perte de revenus.

- Variable cible: `Churn` (`Yes` / `No`)
- Type de problème: classification binaire
- Environnement principal: Databricks + PySpark

## Dataset

Le projet utilise le dataset **Telco Customer Churn**:

- 7043 clients
- 21 variables explicatives + 1 cible (`Churn`)
- Variables démographiques, contractuelles, facturation, services internet/telephonie

## Pipeline du projet

### 1) Chargement et exploration

- Chargement CSV avec schema inféré
- Vérification du schema et des types
- Dictionnaire de données
- Analyse des valeurs manquantes

### 2) Nettoyage des donnees

- Conversion de `TotalCharges` en numerique (`double`)
- Imputation des `null` de `TotalCharges`
- Suppression de `customerID`
- Simplification des modalités (`No internet service` / `No phone service` -> `No`)

### 3) Analyse exploratoire (EDA)

Analyses descriptives et bivariatées sur des hypothèses metier:

- Faible anciennete (`tenure`) -> churn plus élevé
- Contrat `Month-to-month` -> taux de churn tres élevé
- `MonthlyCharges` plus élevés chez les churners

Points observes dans le notebook:

- Taux de churn global: **26.54%**
- Contrat `Month-to-month`: **42.71%** de churn
- Contrat `Two year`: **2.83%** de churn

### 4) Feature engineering

- Creation de la variable `nb_services`
- Encodage de la cible en `label` (0/1)
- Encodage manuel des variables categorielles (binaire / one-hot)
- Assemblage final des variables avec `VectorAssembler`

### 5) Entrainement des modeles

Modèles benchmarkes:

- Regression Logistique
- Random Forest
- Gradient Boosting Trees (GBT)

Split des donnees: **80/20** (train/test)

### 6) Evaluation

Metriques utilisees:

- Precision
- Recall
- F1-score (weighted)

Observations principales notées :

- GBT detecte plus de churners (meilleur rappel sur la classe churn)
- Random Forest est plus precis mais rate davantage de churners
- La Regression Logistique est retenue comme meilleur compromis global (performance, rapidite, interpretabilite)

### 7) Tracking et sauvegarde avec MLflow

Le notebook log:

- Parametres du modele
- Metriques
- Modele Spark final

Puis recharge le modele depuis MLflow pour verification.

## Execution du notebook (Databricks)

1. Importer `Projet_Churn_Databricks.ipynb` dans votre workspace Databricks.
2. Verifier que le fichier de donnees est accessible (adapter `data_path` si besoin).
3. Attacher un cluster avec runtime compatible PySpark/MLflow.
4. Executer les cellules dans l'ordre.

## Librairies principales

- `pyspark`
- `mlflow`
- `pandas`
- `matplotlib`

## Resultat attendu

Une solution de prediction du churn exploitable metier, avec:

- un benchmark de plusieurs algorithmes
- une selection du modele final
- un suivi des exécutions via MLflow
