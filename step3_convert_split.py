"""
Étape 3 — Conversion & Split stratifié (version améliorée)
===========================================================
Améliorations :
  - Split stratifié par densité d'entités (équilibre train/dev)
  - Validation et rapport des spans écartés par cause
  - Affichage de la distribution des labels par split

Usage :
    python step3_convert_split.py --input annotations.json
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import Counter
import spacy
from spacy.tokens import DocBin

def build_docbin(nlp, data: list, path: Path) -> dict:
    db = DocBin()
    report = Counter()
    for text, ann in data:
        doc, ents, occupied = nlp.make_doc(text), [], set()
        for start, end, label in ann["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                report["span_hors_token"] += 1; continue
            if any(t.i in occupied for t in span):
                report["chevauchement"] += 1; continue
            ents.append(span)
            occupied.update(t.i for t in span)
            report[label] += 1
        doc.ents = ents
        db.add(doc)
    db.to_disk(path)
    return dict(report)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     type=Path, default=Path("annotations.json"))
    parser.add_argument("--train-out", type=Path, default=Path("train.spacy"))
    parser.add_argument("--dev-out",   type=Path, default=Path("dev.spacy"))
    parser.add_argument("--seed",      type=int,  default=42)
    args = parser.parse_args()

    nlp  = spacy.blank("en")
    data = json.loads(args.input.read_text(encoding="utf-8"))

    # Split stratifié : articles avec/sans entités distribués proportionnellement
    random.seed(args.seed)
    with_ents    = [d for d in data if d[1]["entities"]]
    without_ents = [d for d in data if not d[1]["entities"]]
    random.shuffle(with_ents)
    random.shuffle(without_ents)

    split_w = int(len(with_ents) * 0.8)
    split_wo = int(len(without_ents) * 0.8)

    train = with_ents[:split_w] + without_ents[:split_wo]
    dev   = with_ents[split_w:] + without_ents[split_wo:]
    random.shuffle(train)
    random.shuffle(dev)

    print(f"[Étape 3] Split stratifié :")
    print(f"  Train : {len(train)} articles ({len(with_ents[:split_w])} avec entités)")
    print(f"  Dev   : {len(dev)} articles ({len(with_ents[split_w:])} avec entités)")

    train_report = build_docbin(nlp, train, args.train_out)
    dev_report   = build_docbin(nlp, dev,   args.dev_out)

    print(f"\n  Distribution train  : {train_report}")
    print(f"  Distribution dev    : {dev_report}")
    print(f"\n[Étape 3] ✅ train.spacy et dev.spacy créés")

if __name__ == "__main__":
    main()
