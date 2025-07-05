# 📈 Time Factor Decay Strategy Framework

## 📌 Présentation

Ce projet implémente un **framework modulaire et réutilisable** pour la construction, l’analyse et le backtesting de **stratégies factorielles**, basé sur le papier de recherche :

> **“Time Factor Information Decay: A Global Study”**  
> *Emlyn Flint & Rademeyer Vermaak*

L’objectif est de tester et comparer différentes approches d’allocation factorielle en tenant compte de la **demi-vie des facteurs**, de la **pureté des expositions**, et du **turnover**.

---

## 🛠️ Fonctionnalités principales

- 🔁 **Backtests multi-périodes et multi-métriques**
- 📊 **Analyse dynamique des expositions factorielles**
- 🧠 **Deux stratégies factorielles implémentées :**

### 1. Portefeuille Factoriel **Pur**
- Exposition nette à un seul facteur
- Rebalancement uniquement lorsque :
  - La **demi-vie** du signal est atteinte
  - L’exposition **non désirée** aux autres facteurs atteint (en absolu) celle du facteur ciblé
- Objectif : **minimiser le turnover** tout en maintenant une exposition pure.

### 2. Portefeuille **Fractile**
- Sélection des titres par **tri factoriel (quantiles)**
- Rebalancement périodique ou conditionnel via demi-vie
- Approche simple mais potentiellement exposée à des facteurs secondaires

---

## 📦 Data Loader Bloomberg

Un module d'importation dédié permet de charger automatiquement :

- Les **prix et fondamentaux** des titres (via Bloomberg API)
- Les **compositions des indices** (tickers)
- Compatible avec de nombreux indices globaux (MSCI, S&P, STOXX...)

---

## 🔬 Demi-vie de l’information factorielle

Le rebalancement s’appuie sur la **demi-vie du facteur** :  
→ Un portefeuille n’est ajusté que lorsque l'exposition à ce facteur atteint la moitié de son exposition initiale.

### Avantages :
- Réduction significative du **turnover**
- Économie de **frais de transaction**
- Maintien d’une **exposition propre et efficace**

---



