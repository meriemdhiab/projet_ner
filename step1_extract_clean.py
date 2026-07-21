"""
Étape 1 — Extraction et Nettoyage (version améliorée)
======================================================
Améliorations vs version originale :
  - Suppression des boilerplates TASS récurrents (signatures, mentions d'agence)
  - Détection et exclusion des articles trop courts ou dupliqués (hash MD5)
  - Conservation des métadonnées utiles (date, tags, link) pour l'analyse downstream
  - Rapport de qualité détaillé en sortie

Sortie : texts_clean.json — liste d'objets {id, text, date, tags, link, char_count}

Usage :
    python step1_extract_clean.py --input data_set.json --output texts_clean.json
"""

from __future__ import annotations
import argparse, json, re, unicodedata, hashlib
from pathlib import Path
from collections import Counter

# ── Patterns de nettoyage ─────────────────────────────────────────────────────
_PARASITE   = re.compile(r"[\u00a0\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_MULTI_SP   = re.compile(r"[ \t]+")
_MULTI_NL   = re.compile(r"\n{3,}")
_HTML_TAG   = re.compile(r"<[^>]+>")

# Boilerplates TASS récurrents à supprimer
_BOILERPLATES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"/TASS/\.?\s*",
        r"MOSCOW,?\s+\w+\s+\d+\.?\s*/TASS/\.?",
        r"©\s*TASS[^.]*\.",
        r"All rights reserved\.?",
        r"Subscribe to TASS[^.]*\.",
        r"Read more[^.]*\.",
        r"\[Photo\][^\n]*",
        r"\[Video\][^\n]*",
    ]
]

_MIN_CHARS = 100  # Longueur minimale d'un texte utile

def clean_text(raw: str) -> str:
    """Nettoyage progressif : HTML → boilerplates → unicode → espaces."""
    text = _HTML_TAG.sub(" ", raw)
    text = unicodedata.normalize("NFC", text)
    text = _PARASITE.sub(" ", text)
    for pat in _BOILERPLATES:
        text = pat.sub("", text)
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()

def fingerprint(text: str) -> str:
    """MD5 des 300 premiers caractères pour détecter les quasi-doublons."""
    return hashlib.md5(text[:300].lower().encode()).hexdigest()

def extract_texts(input_path: Path, min_chars: int = _MIN_CHARS) -> tuple[list, dict]:
    with input_path.open(encoding="utf-8") as f:
        articles = json.load(f)

    records, seen_fps = [], set()
    stats = Counter()

    for article in articles:
        raw = article.get("text") or ""
        cleaned = clean_text(raw)

        if not cleaned:
            stats["vide"] += 1; continue
        if len(cleaned) < min_chars:
            stats["trop_court"] += 1; continue

        fp = fingerprint(cleaned)
        if fp in seen_fps:
            stats["doublon"] += 1; continue
        seen_fps.add(fp)

        records.append({
            "id":         article.get("id"),
            "text":       cleaned,
            "date":       article.get("date"),
            "tags":       article.get("tags", []),
            "link":       article.get("link", ""),
            "char_count": len(cleaned),
        })
        stats["ok"] += 1

    return records, dict(stats)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     type=Path, default=Path("data_set.json"))
    parser.add_argument("--output",    type=Path, default=Path("texts_clean.json"))
    parser.add_argument("--min-chars", type=int,  default=_MIN_CHARS)
    args = parser.parse_args()

    records, stats = extract_texts(args.input, args.min_chars)

    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[Étape 1] ✅ Résultat")
    print(f"  Articles conservés : {stats.get('ok', 0):>6,}")
    print(f"  Vides              : {stats.get('vide', 0):>6,}")
    print(f"  Trop courts        : {stats.get('trop_court', 0):>6,}")
    print(f"  Doublons détectés  : {stats.get('doublon', 0):>6,}")
    print(f"  Fichier            : {args.output}")

if __name__ == "__main__":
    main()
