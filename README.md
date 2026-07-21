# 🎯 OSINT NER — TASS News Analysis

Pipeline complet de **Reconnaissance d'Entités Nommées (NER)** appliqué à l'analyse de la couverture médiatique russe (agence **TASS**) du conflit en Ukraine. Extraction automatique d'entités militaires (armes, organisations, unités) à partir de plus de 21 000 articles de presse, indexation dans **Elasticsearch**, et visualisation via un dashboard **Kibana**.

---

## 📖 Table des matières

- [Contexte & Problématique](#-contexte--problématique)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du projet](#-architecture-du-projet)
- [Stack technique](#-stack-technique)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Pipeline — étape par étape](#-pipeline--étape-par-étape)
- [Résultats & Métriques](#-résultats--métriques)
- [Dashboard Kibana](#-dashboard-kibana)
- [Limites & Améliorations](#-limites--améliorations-possibles)
- [Structure des fichiers](#-structure-des-fichiers)
- [Équipe](#-équipe)

---

## 🧭 Contexte & Problématique

Ce projet OSINT (*Open Source Intelligence*) vise à répondre à la question suivante :

> **Comment extraire automatiquement des informations structurées sur les capacités militaires mentionnées dans la couverture médiatique russe du conflit ukrainien ?**

L'agence de presse **TASS** publie un très grand volume d'articles impossible à dépouiller manuellement. Ce projet met en place un pipeline NLP permettant de :
1. Extraire automatiquement les entités militaires (armes, organisations, unités) de chaque article
2. Structurer ces informations dans un moteur de recherche/analyse (Elasticsearch)
3. Visualiser les tendances et corrélations via un dashboard interactif (Kibana)

---

## ✨ Fonctionnalités

- 🧹 Nettoyage et dédoublonnage d'un corpus de **21 742 articles**
- ✍️ Annotation manuelle d'un échantillon de **200 articles** (labels `WEAPON`, `MIL_ORG`, `MIL_UNIT`)
- 🧠 **Fine-tuning** d'un modèle spaCy (`en_core_web_sm`) sur les données annotées
- 🔎 Inférence NER sur l'ensemble du corpus (**63 553 entités extraites**)
- 🔗 Calcul des co-occurrences entre entités (armes ↔ organisations)
- 📥 Indexation complète dans **Elasticsearch**
- 📊 Dashboard **Kibana** : top armes citées, heatmap de corrélation Armes × Organisations, KPI clés

---

## 🏗️ Architecture du projet

```
Articles TASS bruts (JSON)
        │
        ▼
  ┌───────────────┐
  │   step1        │  Nettoyage (doublons, HTML, articles vides)
  │   clean.py     │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  step2-4       │  Annotation manuelle (200 articles) + split train/dev
  │  annotate.py   │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   step5        │  Fine-tuning spaCy (en_core_web_sm)
  │   finetune.py  │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   step6        │  Inférence NER + fusion + rapport de co-occurrences
  │   merge.py     │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   step7        │  Indexation dans Elasticsearch (index tass_ner_v2)
  │   es_index.py  │
  └───────┬───────┘
          ▼
  ┌────────────────────────────┐
  │  🐳 Docker : Elasticsearch  │
  │           + Kibana          │
  └──────────────┬─────────────┘
                 ▼
       📊 Dashboard Kibana
```

---

## 🛠️ Stack technique

| Composant | Rôle |
|-----------|------|
| **Python 3** | Langage principal du pipeline |
| **spaCy** (`en_core_web_sm`) | Modèle NER fine-tuné pour la reconnaissance d'entités militaires |
| **Elasticsearch 8.11.0** | Indexation et moteur de recherche/agrégation |
| **Kibana 8.11.0** | Visualisation et dashboards interactifs |
| **Docker / docker-compose** | Orchestration et reproductibilité de l'environnement Elasticsearch + Kibana |

---

## ✅ Prérequis

- Python ≥ 3.9
- Docker Desktop (lancé et actif)
- pip / venv (ou tout gestionnaire d'environnement Python)

---

## ⚙️ Installation

### 1. Cloner le repository

```bash
git clone <URL_DU_REPO>
cd ner-tass-osint
```

### 2. Installer les dépendances Python

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ Vérifiez que la version du client `elasticsearch` Python correspond à la version du serveur Elasticsearch utilisée (ex : `pip install "elasticsearch==8.11.0"`), sinon des erreurs de compatibilité peuvent survenir lors de l'indexation.

### 3. Lancer Elasticsearch + Kibana avec Docker

```bash
docker compose up -d
```

Vérifiez que les services sont bien démarrés :

```bash
docker ps
```

Kibana est accessible sur : [http://localhost:5601](http://localhost:5601)
Elasticsearch est accessible sur : [http://localhost:9200](http://localhost:9200)

---

## 🔄 Pipeline — étape par étape

| Script | Commande | Description |
|--------|----------|--------------|
| `step1_clean.py` | `python step1_clean.py` | Nettoie le dataset brut (doublons, HTML, articles vides) |
| `step2-4_annotate.py` | `python step2_annotate.py` | Prépare et structure l'échantillon annoté manuellement |
| `step5_finetune.py` | `python step5_finetune.py` | Fine-tune le modèle spaCy sur les données annotées |
| `step6_merge.py` | `python step6_merge.py` | Applique le modèle sur tout le corpus, génère `dataset_with_entities.json` et le rapport de fusion |
| `step7_elasticsearch.py` | `python step7_elasticsearch.py` | Indexe les documents enrichis dans Elasticsearch (`tass_ner_v2`) |

Exécuter l'ensemble du pipeline dans l'ordre :

```bash
python step1_clean.py
python step5_finetune.py
python step6_merge.py
python step7_elasticsearch.py
```

Une fois l'indexation terminée, créez la **Data View** `tass_ner_v2` dans Kibana (`Stack Management → Data Views`) pour accéder aux visualisations.

---

## 📊 Résultats & Métriques

### Statistiques du dataset
- **21 742** articles indexés
- **63 553** entités militaires extraites (≈2,9 entités/article)
- **87,1%** des articles contiennent au moins une entité militaire détectée
- **1 059** formulations textuelles distinctes d'armes détectées

### Performance du modèle (F1-score)
| Label | F1-score | Occurrences |
|-------|----------|--------------|
| **Global** | **~79,4%** | — |
| `MIL_ORG` | ~85% | 45 877 |
| `WEAPON` | ~67% | 17 638 |
| `MIL_UNIT` | 0.000 ⚠️ | 37 (données insuffisantes) |

### Top résultats observés
- Armes les plus citées : **cruise missiles** (~1030), **S-400** (~730), **ballistic missile**, **air defense missile**
- Co-occurrences fortes : `Kalibr × Russian Navy` (258), `S-400 × NATO` (244), `ballistic missiles × NATO` (204)

---

## 📈 Dashboard Kibana

Le dashboard final regroupe :

1. **Top 10 armes citées** (Bar chart) — met en évidence la domination des missiles de croisière et systèmes de défense antiaérienne
2. **Heatmap Armes × Organisations militaires** — révèle des associations narratives distinctes (ex : HIMARS ↔ ministère russe de la Défense, cruise missiles ↔ NATO)
3. **KPI clés** — Articles indexés, Entités extraites, Taux de couverture

> Le dashboard permet d'aller au-delà d'un simple comptage : il révèle des patterns de discours et des associations sémantiques entre armes et acteurs militaires.

---

## ⚠️ Limites & Améliorations possibles

- **Déséquilibre des classes** : `MIL_UNIT` est fortement sous-représenté dans les données d'entraînement (26 exemples), ce qui explique un F1 de 0 sur ce label
- **Sur-segmentation des entités** : certaines variantes textuelles d'une même entité réelle sont comptées séparément faute de normalisation complète (ex : "Defense" vs "the Russian Defense Ministry")
- **Biais de source unique** : le corpus provient exclusivement de TASS (agence de presse officielle russe), ce qui reflète un narratif spécifique

**Pistes d'amélioration :**
- Annoter davantage d'exemples pour `MIL_UNIT`
- Ajouter une étape de normalisation/résolution de coréférence
- Étendre le corpus à d'autres sources médiatiques pour comparer les narratifs
- Automatiser le pipeline complet (orchestration via Airflow ou équivalent)

---

## 📂 Structure des fichiers

```
ner-tass-osint/
├── data/
│   ├── data_set.json                  # Articles bruts
│   └── dataset_with_entities.json     # Articles enrichis après inférence NER
├── step1_clean.py                     # Nettoyage du corpus
├── step2_annotate.py                  # Préparation de l'annotation manuelle
├── step5_finetune.py                  # Fine-tuning du modèle spaCy
├── step6_merge.py                     # Inférence + fusion + rapport de co-occurrences
├── step7_elasticsearch.py             # Indexation dans Elasticsearch
├── docker-compose.yml                 # Orchestration Elasticsearch + Kibana
├── requirements.txt                   # Dépendances Python
└── README.md
```

---

## 👥 Équipe

- **Meriem Dhiab**
- **Shaineze Menzel**
- **Thierry Alvares**

---

*Projet réalisé dans le cadre du cursus MSc — Data Pipelines for AI / OSINT NER, 2025-2026.*
