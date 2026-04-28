# Analyse Comportementale Clientèle Retail

## Description du projet

Ce projet est réalisé dans le cadre d’un atelier de **Machine Learning**.
L’objectif est d’analyser le comportement des clients d’une entreprise e-commerce de cadeaux afin de mieux comprendre leurs profils, prédire le **churn** et proposer des recommandations marketing.

## Objectifs

- Explorer la qualité et la structure des données.
- Nettoyer les valeurs manquantes et aberrantes.
- Encoder les variables catégorielles.
- Normaliser les variables numériques.
- Réduire la dimension avec l’ACP (PCA).
- Construire des modèles de classification, clustering et régression.
- Déployer le modèle avec Flask.

## Structure du projet

```text
projet_ml_retail/
├── data/
│   ├── raw/
│   ├── processed/
│   └── train_test/
├── notebooks/
├── src/
├── models/
├── app/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

> **Note :** dans cette version du dépôt, le fichier de dépendances est actuellement nommé `requirements`.

## Dataset

Le dataset contient **4 372 clients** et **52 variables** décrivant les transactions, les comportements, les segments clients et le churn.

La variable cible principale est :

- **Churn**
  - `0` : client fidèle
  - `1` : client parti

## Étape 8 — Premier test GitHub

Dans le terminal :

```bash
git init
git add .
git commit -m "Initial project structure"
```
