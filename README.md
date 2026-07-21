# 🎯 TASS NER Pipeline — Groupe (version améliorée)

Pipeline NER militaire sur 21 742 articles TASS, avec extraction de 3 labels :
`WEAPON` | `MIL_UNIT` | `MIL_ORG`

## Améliorations vs version de référence

| Étape | Amélioration |
|-------|-------------|
| Step 1 | Suppression boilerplates TASS, déduplication par hash MD5 |
| Step 2 | Validation contextuelle, normalisation canonique, sample stratifié |
| Step 3 | Split stratifié par densité d'entités |
| Step 4 | Config optimisée : Adam + warmup + early stopping, éval par label |
| Step 5 | Filtrage confiance, normalisation, déduplication intra-article |
| Step 6 | Fusion par ID (O(1)), co-occurrences WEAPON × MIL_ORG |
| Step 7 | `dominant_label`, `entity_density`, `timeline_month`, requêtes de test |

## Lancement (Google Colab)
1. Ouvre `PIPELINE_COLAB.ipynb` sur https://colab.research.google.com
2. Active GPU : Exécution → T4 GPU
3. Upload `data_set.json` + tous les `step*.py`
4. Exécute les cellules dans l'ordre

## Lancement (local)
```bash
pip install 'spacy>=3.7' elasticsearch
python -m spacy download en_core_web_sm
python step1_extract_clean.py
python step2_spacy_annotate.py
python step3_convert_split.py
python step4_train_model.py
python step5_inference.py
python step6_data_merged.py
docker-compose up -d
python step7_elasticsearch.py --force
```
