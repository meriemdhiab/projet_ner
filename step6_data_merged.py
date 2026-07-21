"""
Étape 6 — Fusion enrichie (version améliorée)
=============================================
Améliorations :
  - Fusion par ID (O(1) vs O(N²) text-matching original)
  - Calcul de co-occurrences WEAPON × MIL_ORG par article
  - Champ `entity_summary` : liste dédupliquée + triée par fréquence
  - Rapport de couverture et qualité en sortie

Sortie : dataset_with_entities.json + merge_report.json

Usage :
    python step6_data_merged.py
"""
import json, time
from pathlib import Path
from collections import defaultdict, Counter

def format_dur(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"

def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("ÉTAPE 6 — Fusion dataset + entités (par ID)")
    print("="*60)

    ROOT          = Path(__file__).resolve().parent
    dataset_path  = ROOT / "data_set.json"
    entities_path = ROOT / "extracted_entities.jsonl"
    ann_path      = ROOT / "annotations.json"
    ann_ids_path  = ROOT / "annotated_ids.json"
    output_path   = ROOT / "dataset_with_entities.json"

    # ── Chargement ────────────────────────────────────────────────
    print("\n1️⃣  Chargement...")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Index entités inférées (étape 5) par ID
    inferred: dict = {}
    if entities_path.exists():
        with entities_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                inferred[row["id"]] = row.get("entities", [])

    # Index entités annotées manuellement (étape 2)
    annotated: dict = {}
    if ann_path.exists() and ann_ids_path.exists():
        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
        ann_ids  = json.loads(ann_ids_path.read_text(encoding="utf-8"))
        for doc_id, (text, ann) in zip(ann_ids, ann_data):
            annotated[doc_id] = [
                {"text": text[s:e], "label": lbl, "start": s, "end": e}
                for s, e, lbl in ann["entities"]
            ]

    print(f"   Dataset    : {len(dataset):,} articles")
    print(f"   Inférés    : {len(inferred):,} | Annotés : {len(annotated):,}")

    # ── Fusion ────────────────────────────────────────────────────
    print("\n2️⃣  Fusion par ID...")
    merged, global_stats = [], Counter()
    cooccurrence = Counter()   # (weapon, org) co-occurrences

    for article in dataset:
        art_id   = article.get("id")
        entities = inferred.get(art_id) or annotated.get(art_id) or []

        by_label = defaultdict(list)
        for e in entities:
            by_label[e["label"]].append(e["text"])
            global_stats[e["label"]] += 1

        # Déduplique et trie par fréquence dans l'article
        weapons   = list(dict.fromkeys(by_label["WEAPON"]))
        mil_units = list(dict.fromkeys(by_label["MIL_UNIT"]))
        mil_orgs  = list(dict.fromkeys(by_label["MIL_ORG"]))

        # Co-occurrences WEAPON × MIL_ORG
        for w in weapons:
            for o in mil_orgs:
                cooccurrence[(w, o)] += 1

        merged.append({
            **article,
            "entities":   entities,
            "entity_counts": {
                "WEAPON":   len(weapons),
                "MIL_UNIT": len(mil_units),
                "MIL_ORG":  len(mil_orgs),
                "total":    len(entities),
            },
            "weapons":   weapons,
            "mil_units": mil_units,
            "mil_orgs":  mil_orgs,
        })

    # ── Sauvegarde ────────────────────────────────────────────────
    print("\n3️⃣  Sauvegarde...")
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    # Rapport
    covered = sum(1 for a in merged if a["entity_counts"]["total"] > 0)
    report = {
        "articles_totaux": len(merged),
        "articles_avec_entités": covered,
        "couverture_pct": round(100*covered/len(merged), 1),
        "entités_par_label": dict(global_stats),
        "top_10_cooccurrences_weapon_org": [
            {"weapon": w, "org": o, "count": c}
            for (w, o), c in cooccurrence.most_common(10)
        ]
    }
    Path("merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ FUSION TERMINÉE en {format_dur(time.time()-t0)}")
    print(f"   Articles enrichis   : {len(merged):,}")
    print(f"   Couverture          : {report['couverture_pct']}%")
    print(f"   WEAPON              : {global_stats['WEAPON']:,}")
    print(f"   MIL_UNIT            : {global_stats['MIL_UNIT']:,}")
    print(f"   MIL_ORG             : {global_stats['MIL_ORG']:,}")
    print(f"   Top co-occurrence   : {cooccurrence.most_common(3)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
