"""
Étape 7 — Indexation Elasticsearch enrichie (version améliorée)
================================================================
Améliorations vs version originale :
  - Champs keyword dénormalisés pour agrégations Kibana rapides
  - Champ `dominant_label` (label le plus représenté par article)
  - Champ `entity_density` (entités / 1000 caractères) pour filtres
  - Champ `timeline_month` (YYYY-MM) pour histogrammes temporels
  - Vérification de santé ES avant indexation
  - Rapport post-indexation avec requêtes de test

Usage :
    python step7_elasticsearch.py --input dataset_with_entities.json --force
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from collections import Counter

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    print("❌ pip install elasticsearch"); sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("elasticsearch_pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("ES_Pipeline")

INDEX_NAME = "tass_ner_v2"

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1, "number_of_replicas": 0,
        "analysis": {"analyzer": {"entity_analyzer": {
            "type": "custom", "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding"]
        }}}
    },
    "mappings": {"properties": {
        # Métadonnées
        "article_id":      {"type": "keyword"},
        "title":           {"type": "text", "analyzer": "standard",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
        "text":            {"type": "text", "analyzer": "standard"},
        "url":             {"type": "keyword"},
        "date":            {"type": "date",
                            "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd||dd.MM.yyyy"},
        "timeline_month":  {"type": "keyword"},   # YYYY-MM pour histogrammes
        "tags":            {"type": "keyword"},

        # Entités (nested = requêtes complexes possibles)
        "entities": {"type": "nested", "properties": {
            "text":  {"type": "text",
                      "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            "label": {"type": "keyword"},
            "start": {"type": "integer"},
            "end":   {"type": "integer"},
        }},

        # Compteurs (agrégations rapides)
        "entity_counts": {"properties": {
            "WEAPON":   {"type": "integer"},
            "MIL_UNIT": {"type": "integer"},
            "MIL_ORG":  {"type": "integer"},
            "total":    {"type": "integer"},
        }},

        # Champs dénormalisés pour facettes Kibana
        "weapons":          {"type": "keyword"},
        "mil_units":        {"type": "keyword"},
        "mil_orgs":         {"type": "keyword"},
        "all_entity_texts": {"type": "text", "analyzer": "entity_analyzer"},

        # Champs analytiques additionnels
        "dominant_label":   {"type": "keyword"},   # label le plus fréquent
        "entity_density":   {"type": "float"},      # entités / 1000 chars
    }}
}

def dominant_label(counts: dict) -> str:
    labels = {k: v for k, v in counts.items() if k != "total" and v > 0}
    return max(labels, key=labels.get) if labels else "none"

def to_epoch_millis(date_raw) -> int | None:
    """Le champ `date` de data_set.json est un timestamp Unix EN SECONDES
    (ex: 1761470423 = octobre 2025). Le mapping ES attend des millisecondes
    (epoch_millis), donc on multiplie par 1000 pour éviter que tout tombe
    autour du 1er janvier 1970."""
    if date_raw is None:
        return None
    try:
        return int(float(date_raw) * 1000)
    except (TypeError, ValueError):
        return None

def timeline_month(epoch_ms: int | None) -> str | None:
    if not epoch_ms: return None
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
    except Exception:
        return None

def generate_docs(data: list):
    for art in data:
        ent_counts  = art.get("entity_counts", {})
        text        = art.get("text", "")
        char_count  = max(len(text), 1)
        total_ents  = ent_counts.get("total", 0)
        all_texts   = " ".join(e.get("text","") for e in art.get("entities",[]))
        date_ms     = to_epoch_millis(art.get("date"))

        yield {
            "_index": INDEX_NAME,
            "_id":    str(art.get("id")),
            "_source": {
                "article_id":     str(art.get("id")),
                "title":          art.get("title", ""),
                "text":           text[:10000],
                "url":            art.get("link", ""),
                "date":           date_ms,
                "timeline_month": timeline_month(date_ms),
                "tags":           art.get("tags", []),
                "entities":       art.get("entities", []),
                "entity_counts":  ent_counts,
                "weapons":        art.get("weapons", []),
                "mil_units":      art.get("mil_units", []),
                "mil_orgs":       art.get("mil_orgs", []),
                "all_entity_texts": all_texts,
                "dominant_label": dominant_label(ent_counts),
                "entity_density": round(total_ents / char_count * 1000, 3),
            }
        }

def run_test_queries(es: Elasticsearch) -> None:
    """Quelques requêtes de validation post-indexation."""
    logger.info("\n🔍 Requêtes de validation :")
    # Comptage global
    total = es.count(index=INDEX_NAME)["count"]
    logger.info(f"  Total articles indexés : {total:,}")
    # Top armes
    res = es.search(index=INDEX_NAME, body={
        "size": 0,
        "aggs": {"top_weapons": {"terms": {"field": "weapons", "size": 5}}}
    })
    top = [(b["key"], b["doc_count"]) for b in res["aggregations"]["top_weapons"]["buckets"]]
    logger.info(f"  Top 5 WEAPON : {top}")
    # Top orgs
    res = es.search(index=INDEX_NAME, body={
        "size": 0,
        "aggs": {"top_orgs": {"terms": {"field": "mil_orgs", "size": 5}}}
    })
    top = [(b["key"], b["doc_count"]) for b in res["aggregations"]["top_orgs"]["buckets"]]
    logger.info(f"  Top 5 MIL_ORG : {top}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset_with_entities.json"))
    parser.add_argument("--host",  default="localhost")
    parser.add_argument("--port",  type=int, default=9200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    es = Elasticsearch([{"host": args.host, "port": args.port, "scheme": "http"}], request_timeout=30)
    if not es.ping():
        logger.error(f"Elasticsearch inaccessible sur {args.host}:{args.port}")
        logger.error("Lance : docker-compose up -d")
        raise SystemExit(1)

    info = es.info()
    logger.info(f"✓ Elasticsearch v{info['version']['number']}")

    if es.indices.exists(index=INDEX_NAME):
        if args.force:
            es.indices.delete(index=INDEX_NAME)
            logger.info(f"Index '{INDEX_NAME}' recréé")
        else:
            logger.info(f"Index '{INDEX_NAME}' existant (--force pour recréer)")
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        logger.info(f"✓ Index '{INDEX_NAME}' créé")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    logger.info(f"Indexation de {len(data):,} articles...")

    t = time.time()
    ok, failed = helpers.bulk(es, generate_docs(data), chunk_size=500, raise_on_error=False)
    logger.info(f"✅ {ok:,} indexés en {time.time()-t:.1f}s | {len(failed) if failed else 0} erreurs")

    run_test_queries(es)
    logger.info(f"\n🌐 Kibana → http://localhost:5601")
    logger.info(f"   Index à créer dans Kibana : '{INDEX_NAME}'")

if __name__ == "__main__":
    main()