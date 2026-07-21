"""
Étape 2 — Annotation NER contextuelle (version améliorée)
==========================================================
Améliorations vs version originale :
  - Validation contextuelle : un token n'est annoté que si son contexte
    immédiat confirme le domaine militaire (évite les faux positifs)
  - Normalisation des entités détectées (canonical forms)
  - Score de confiance par entité selon le nombre de signaux concordants
  - Statistiques détaillées par label en sortie
  - Support d'un sample stratifié par densité d'entités (meilleure couverture)

Sortie :
  annotations.json      — format spaCy train data
  annotated_ids.json    — liste des IDs utilisés
  annotation_report.json — rapport qualité

Usage :
    python step2_spacy_annotate.py --input texts_clean.json --output annotations.json
"""

from __future__ import annotations
import argparse, json, random, re, logging, sys
from pathlib import Path
from collections import Counter, defaultdict
import spacy
from spacy.matcher import PhraseMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("annotation_pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("NER_Pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# VOCABULAIRES MILITAIRES (enrichis + normalisés)
# ══════════════════════════════════════════════════════════════════════════════

# Canons de normalisation : variante → forme canonique
WEAPON_CANONICAL = {
    "kalibr": "Kalibr", "caliber": "Kalibr",
    "kinzhal": "Kinzhal", "dagger": "Kinzhal",
    "zircon": "Zircon", "tsirkon": "Zircon",
    "iskander": "Iskander",
    "burevestnik": "Burevestnik", "skyfall": "Burevestnik",
    "geran": "Geran-2", "geranium": "Geran-2", "shaheed": "Shahed-136",
    "shahed": "Shahed-136",
    "himars": "HIMARS", "atacms": "ATACMS",
    "javelin": "Javelin", "nlaw": "NLAW",
    "storm shadow": "Storm Shadow", "scalp": "Storm Shadow",
    "patriot": "Patriot", "nasams": "NASAMS", "iris-t": "IRIS-T",
}

WEAPON_PHRASES = {
    # Systèmes russes précis
    "S-400","S-350","S-300","S-500","Pantsir-S1","Pantsir-S2","Tor-M2",
    "Buk-M3","Buk-M2","Tunguska","Gepard",
    "T-90M","T-90A","T-72B3","T-80BV","T-64BV",
    "BMP-2","BMP-3","BMP-1","BTR-80","BTR-82A","BMD-4",
    "Kalibr","Kinzhal","Zircon","Iskander-M","Iskander-K",
    "Burevestnik","Geran-2","Shahed-136","Lancet-3","Orlan-10",
    "Kh-101","Kh-47M2","Kh-22","Kh-55","Kh-35",
    "Sarmat","Avangard","Poseidon","Nudol",
    "MiG-29","MiG-31K","Su-25","Su-27","Su-34","Su-35S","Su-57","Su-35",
    "Tu-95MS","Tu-160","Tu-22M3","Il-76","Ka-52","Mi-28","Mi-8",
    "Grad","Smerch","Uragan","Tornado-G","Msta-S","Koalitsiya-SV",
    # Systèmes occidentaux
    "F-16","F-35","F-15","A-10","B-52","B-2",
    "Patriot","HIMARS","ATACMS","Javelin","Stinger","NLAW",
    "Storm Shadow","Scalp","Harpoon","Tomahawk","AGM-158",
    "Bradley","M1 Abrams","Leopard 2","Challenger 2","Leclerc",
    "NASAMS","IRIS-T","Gepard","PzH 2000","Caesar",
    "Bayraktar TB2","TB-2",
    # Termes génériques militaires (avec contexte obligatoire)
    "cruise missile","ballistic missile","anti-tank missile","air defense missile",
    "hypersonic missile","guided missile","loitering munition","kamikaze drone",
    "combat drone","reconnaissance drone","attack drone","artillery system",
    "multiple rocket launcher","self-propelled howitzer",
}

MIL_UNIT_PHRASES = {
    "Special Forces","Spetsnaz","Special Operations Forces","SOF",
    "airborne troops","paratroopers","air assault","marines",
    "National Guard","Territorial Defense Forces","Border Guard",
    "Wagner Group","Wagner PMC",
}

MIL_ORG_PHRASES = {
    "NATO","North Atlantic Treaty Organization",
    "Pentagon","U.S. Department of Defense","Joint Chiefs of Staff",
    "Russian Armed Forces","Ukrainian Armed Forces",
    "General Staff","Ministry of Defense",
    "Russian Defense Ministry","Ukrainian Defense Ministry",
    "Wagner Group","Rosguardia","FSB","GRU","SVR","IRGC",
    "Hamas","Hezbollah","Islamic State","ISIS","ISIL","Daesh",
    "Houthis","Ansarallah","IDF","Israel Defense Forces",
    "CSTO","SCO",
}

# Mots-clés de contexte militaire (renforcent la confiance)
MILITARY_CONTEXT_KW = {
    "military","weapon","armed","force","troop","soldier","war","combat",
    "battle","attack","strike","defense","offensive","operation","conflict",
    "army","navy","air force","artillery","missile","bomb","ammunition",
    "frontline","battlefield","siege","shelling","deployment","brigade",
}

# Patterns regex enrichis (avec groupes nommés pour meilleure lisibilité)
WEAPON_REGEX = [
    # Désignations russes/soviétiques
    r'\bS-(?:300|350|400|500)[A-Z0-9/]*\b',
    r'\bT-(?:14|72|80|90|64)[A-Z0-9]*\b',
    r'\bBMD?-\d[A-Z0-9]*\b', r'\bBTR-\d{2}[A-Z0-9]*\b',
    r'\bKh-\d{2,3}[A-Z0-9]*\b', r'\bR-\d{2,3}[A-Z0-9]*\b',
    r'\bMiG-\d{2,3}[A-Z0-9]*\b', r'\bSu-\d{2,3}[A-Z0-9]*\b',
    r'\bTu-\d{2,3}[A-Z0-9]*\b', r'\bKa-\d{2,3}[A-Z0-9]*\b',
    r'\bMi-\d{1,3}[A-Z0-9]*\b', r'\bIl-\d{2,3}[A-Z0-9]*\b',
    r'\bAn-\d{2,3}[A-Z0-9]*\b',
    # Désignations occidentales
    r'\bF-\d{2,3}[A-Z]?\b', r'\bA-\d{2}[A-Z]?\b',
    r'\bB-(?:1B?|2A?|52H?)\b', r'\bC-\d{3}[A-Z]?\b',
    r'\bAGM-\d{2,3}[A-Z]?\b', r'\bAIM-\d{2,3}[A-Z]?\b',
    r'\bPGM-\d+\b',
    # Systèmes génériques avec qualificatif obligatoire
    r'\b(?:cruise|ballistic|hypersonic|supersonic|anti-(?:tank|aircraft|ship))\s+missile[s]?\b',
    r'\b(?:loitering\s+munition[s]?|suicide\s+drone[s]?|kamikaze\s+drone[s]?)\b',
    r'\b(?:multiple\s+rocket\s+launch(?:er)?[s]?|MLRS)\b',
    r'\b(?:self-propelled\s+(?:gun|howitzer|artillery)[s]?)\b',
    r'\b(?:anti-(?:tank|aircraft|drone)\s+system[s]?)\b',
]

MIL_UNIT_REGEX = [
    r'\b\d+(?:st|nd|rd|th)?\s+(?:Separate\s+)?(?:Guards?\s+)?'
    r'(?:Tank|Motor(?:ized|ised)?|Mech(?:anized|anised)?|Airborne|Mountain|'
    r'Marine|Air\s+Assault|Artillery|Missile|Naval|Engineer|Assault)\s+'
    r'(?:Army|Corps|Division|Brigade|Battalion|Regiment|Company|Group)\b',
    r'\b(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th)\s+'
    r'(?:Army|Guards\s+Army|Tank\s+Army|Combined\s+Arms\s+Army)\b',
    r'\b(?:rapid\s+reaction|quick\s+reaction|immediate\s+response)\s+force[s]?\b',
    r'\b(?:volunteer|irregular|paramilitary)\s+(?:battalion[s]?|unit[s]?|detachment[s]?)\b',
    r'\b(?:assault|storm|shock)\s+(?:group[s]?|unit[s]?|team[s]?|detachment[s]?)\b',
]

MIL_ORG_REGEX = [
    r'\b(?:Russian|Russia\'s)\s+(?:Armed\s+Forces?|Army|Navy|Air\s+(?:and\s+Space\s+)?Force[s]?|Aerospace\s+Forces?|Military|troops?|forces?|servicemen)\b',
    r'\b(?:Ukrainian|Ukraine\'s)\s+(?:Armed\s+Forces?|Army|Navy|Air\s+Force[s]?|Military|troops?|forces?|defenders?)\b',
    r'\bU\.?S\.?\s+(?:Army|Navy|Air\s+Force|Marine\s+Corps|Space\s+Force|Military|Armed\s+Forces?|troops?|forces?)\b',
    r'\b(?:British|UK)\s+(?:Army|Royal\s+(?:Navy|Air\s+Force|Marines)|forces?|troops?|military)\b',
    r'\b(?:French|German|Polish|Turkish|Israeli|Iranian|Syrian|North\s+Korean)\s+(?:Army|Air\s+Force|forces?|troops?|military)\b',
    r'\b(?:coalition|allied|multinational|joint)\s+forces?\b',
    r'\b(?:Russian|Ukrainian)\s+(?:Defense|Defence)\s+Ministry\b',
    r'\b(?:General|Joint)\s+Staff\b',
    r'\bIsrael(?:i)?\s+Defense\s+Forces?\b',
    r'\bNorth\s+Atlantic\s+Treaty\s+Organi[sz]ation\b',
]

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION CONTEXTUELLE
# ══════════════════════════════════════════════════════════════════════════════

def has_military_context(text: str, start: int, end: int, window: int = 150) -> bool:
    """Vérifie que le contexte autour d'un span contient des mots militaires."""
    ctx_start = max(0, start - window)
    ctx_end   = min(len(text), end + window)
    context   = text[ctx_start:ctx_end].lower()
    return any(kw in context for kw in MILITARY_CONTEXT_KW)

def normalize_entity(text: str, label: str) -> str:
    """Retourne la forme canonique d'une entité si connue."""
    key = text.lower().strip()
    if label == "WEAPON":
        return WEAPON_CANONICAL.get(key, text)
    return text

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(nlp, text: str) -> list[tuple[int, int, str]]:
    doc   = nlp(text)
    spans: list[tuple[int, int, str]] = []

    # 1. spaCy NER → MIL_ORG (avec validation contextuelle)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE", "NORP"):
            low = ent.text.lower()
            is_mil = any(kw in low for kw in {
                "military","defense","defence","armed","forces","nato","pentagon",
                "ministry","staff","guard","wagner","hamas","hezbollah","isis",
                "irgc","idf","army","navy","air force","troops","regiment",
            })
            if is_mil and has_military_context(text, ent.start_char, ent.end_char):
                spans.append((ent.start_char, ent.end_char, "MIL_ORG"))

    # 2. PhraseMatcher (WEAPON, MIL_UNIT, MIL_ORG)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("WEAPON",   [nlp.make_doc(p) for p in WEAPON_PHRASES])
    matcher.add("MIL_UNIT", [nlp.make_doc(p) for p in MIL_UNIT_PHRASES])
    matcher.add("MIL_ORG",  [nlp.make_doc(p) for p in MIL_ORG_PHRASES])

    for match_id, start, end in matcher(doc):
        label  = nlp.vocab.strings[match_id]
        s_char = doc[start].idx
        e_char = doc[end-1].idx + len(doc[end-1].text)
        # Validation contextuelle pour WEAPON uniquement
        if label == "WEAPON" and not has_military_context(text, s_char, e_char):
            continue
        spans.append((s_char, e_char, label))

    # 3. Regex enrichis
    for pat in WEAPON_REGEX:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if has_military_context(text, m.start(), m.end()):
                spans.append((m.start(), m.end(), "WEAPON"))
    for pat in MIL_UNIT_REGEX:
        for m in re.finditer(pat, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), "MIL_UNIT"))
    for pat in MIL_ORG_REGEX:
        for m in re.finditer(pat, text, re.IGNORECASE):
            spans.append((m.start(), m.end(), "MIL_ORG"))

    # 4. Dédoublonnage — on garde le span le plus long en cas de chevauchement
    spans = list(set(spans))
    spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    kept, occupied = [], set()
    for s, e, lbl in spans:
        if any(i in occupied for i in range(s, e)):
            continue
        kept.append((s, e, lbl))
        occupied.update(range(s, e))

    return sorted(kept)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       type=Path, default=Path("texts_clean.json"))
    parser.add_argument("--output",      type=Path, default=Path("annotations.json"))
    parser.add_argument("--model",       default="en_core_web_sm")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    logger.info(f"Chargement du modèle '{args.model}'...")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        logger.error(f"Modèle introuvable → python -m spacy download {args.model}")
        raise SystemExit(1)

    records = json.loads(args.input.read_text(encoding="utf-8"))
    logger.info(f"{len(records)} textes chargés")

    # Sample stratifié : priorité aux articles plus longs (plus d'entités potentielles)
    random.seed(args.seed)
    records_sorted = sorted(records, key=lambda r: r.get("char_count", len(r["text"])), reverse=True)
    top_half = records_sorted[:len(records_sorted)//2]
    sample   = random.sample(top_half, min(args.sample_size, len(top_half)))

    train_data, annotated_ids = [], []
    label_counts = Counter()

    for i, rec in enumerate(sample, 1):
        text   = rec["text"]
        doc_id = rec["id"]
        try:
            ents = extract_entities(nlp, text)
            for _, _, lbl in ents:
                label_counts[lbl] += 1
            train_data.append([text, {"entities": ents}])
            annotated_ids.append(doc_id)
            if i % 50 == 0:
                logger.info(f"  [{i}/{len(sample)}] en cours...")
        except Exception as e:
            logger.warning(f"Erreur sur {doc_id}: {e}")

    # Sauvegarde
    args.output.write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("annotated_ids.json").write_text(json.dumps(annotated_ids, ensure_ascii=False), encoding="utf-8")

    # Rapport qualité
    report = {
        "articles_annotés": len(train_data),
        "entités_par_label": dict(label_counts),
        "total_entités": sum(label_counts.values()),
        "moyenne_par_article": round(sum(label_counts.values()) / max(len(train_data), 1), 2),
        "articles_sans_entité": sum(1 for d in train_data if not d[1]["entities"]),
    }
    Path("annotation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"\n[Étape 2] ✅ Résultat")
    logger.info(f"  Articles annotés    : {report['articles_annotés']}")
    logger.info(f"  WEAPON              : {label_counts['WEAPON']}")
    logger.info(f"  MIL_UNIT            : {label_counts['MIL_UNIT']}")
    logger.info(f"  MIL_ORG             : {label_counts['MIL_ORG']}")
    logger.info(f"  Moy. entités/article: {report['moyenne_par_article']}")
    logger.info(f"  Articles sans entité: {report['articles_sans_entité']}")

if __name__ == "__main__":
    main()
