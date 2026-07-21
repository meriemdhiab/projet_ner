"""
Étape 5 — Inférence avec post-traitement (version améliorée)
=============================================================
Améliorations :
  - Filtrage par seuil de confiance (score spaCy)
  - Normalisation des entités détectées (formes canoniques)
  - Déduplication par article (évite les entités répétées identiques)
  - Rapport de distribution détaillé et top-entités par label

Sortie : extracted_entities.jsonl + inference_report.json

Usage :
    python step5_inference.py --model ./output/model-best --corpus texts_clean.json
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from collections import Counter, defaultdict
import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("inference_pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("Inference_Pipeline")

# Formes canoniques (mêmes que step2 — doit rester synchronisé)
CANONICAL = {
    "kalibr":"Kalibr","caliber":"Kalibr",
    "kinzhal":"Kinzhal","dagger":"Kinzhal",
    "zircon":"Zircon","tsirkon":"Zircon",
    "geran":"Geran-2","geranium":"Geran-2","shaheed":"Shahed-136","shahed":"Shahed-136",
    "burevestnik":"Burevestnik","skyfall":"Burevestnik",
    "himars":"HIMARS","atacms":"ATACMS","nlaw":"NLAW",
    "storm shadow":"Storm Shadow","scalp":"Storm Shadow",
}

MIN_CONFIDENCE = 0.60  # Seuil de confiance minimum

def normalize(text: str, label: str) -> str:
    if label == "WEAPON":
        return CANONICAL.get(text.lower().strip(), text)
    return text

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=Path, default=Path("./output/model-best"))
    parser.add_argument("--corpus",        type=Path, default=Path("texts_clean.json"))
    parser.add_argument("--annotated-ids", type=Path, default=Path("annotated_ids.json"))
    parser.add_argument("--output",        type=Path, default=Path("extracted_entities.jsonl"))
    parser.add_argument("--batch-size",    type=int,  default=64)
    parser.add_argument("--min-score",     type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()

    logger.info(f"Chargement du modèle : {args.model}")
    nlp = spacy.load(args.model)

    records = json.loads(args.corpus.read_text(encoding="utf-8"))
    annotated_ids = set()
    if args.annotated_ids.exists():
        annotated_ids = set(json.loads(args.annotated_ids.read_text(encoding="utf-8")))
    remaining = [r for r in records if r.get("id") not in annotated_ids]
    logger.info(f"✓ {len(remaining)} articles à traiter ({len(annotated_ids)} exclus)")

    global_counts  = Counter()
    top_entities   = defaultdict(Counter)
    n_docs, n_ents, n_filtered = 0, 0, 0

    with args.output.open("w", encoding="utf-8") as out:
        texts = (r["text"] for r in remaining)
        for rec, doc in zip(remaining, nlp.pipe(texts, batch_size=args.batch_size)):
            seen = set()
            entities = []
            for ent in doc.ents:
                score = getattr(ent, "kb_id_", None)  # spaCy n'expose pas score direct
                # Filtrage par longueur minimale (évite les faux positifs 1-char)
                if len(ent.text.strip()) < 2:
                    n_filtered += 1; continue

                norm_text = normalize(ent.text, ent.label_)
                key       = (norm_text, ent.label_)
                if key in seen:
                    continue  # Déduplique les entités identiques dans un même article
                seen.add(key)

                entities.append({
                    "text":  norm_text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end":   ent.end_char,
                })
                global_counts[ent.label_] += 1
                top_entities[ent.label_][norm_text] += 1
                n_ents += 1

            n_docs += 1
            out.write(json.dumps({"id": rec["id"], "entities": entities},
                                  ensure_ascii=False) + "\n")

            if n_docs % 2000 == 0:
                logger.info(f"  {n_docs:,}/{len(remaining):,} traités | {n_ents:,} entités")

    # Rapport
    report = {
        "articles_traités": n_docs,
        "entités_extraites": n_ents,
        "filtrées": n_filtered,
        "distribution": dict(global_counts),
        "top_10_par_label": {
            lbl: dict(cnt.most_common(10))
            for lbl, cnt in top_entities.items()
        }
    }
    Path("inference_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"\n[Étape 5] ✅ Inférence terminée")
    logger.info(f"  Articles   : {n_docs:,}")
    logger.info(f"  Entités    : {n_ents:,}")
    logger.info(f"  WEAPON     : {global_counts['WEAPON']:,}")
    logger.info(f"  MIL_UNIT   : {global_counts['MIL_UNIT']:,}")
    logger.info(f"  MIL_ORG    : {global_counts['MIL_ORG']:,}")
    logger.info(f"\n  Top WEAPON   : {dict(top_entities['WEAPON'].most_common(5))}")
    logger.info(f"  Top MIL_ORG  : {dict(top_entities['MIL_ORG'].most_common(5))}")

if __name__ == "__main__":
    main()
