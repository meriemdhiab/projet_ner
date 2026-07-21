"""
Étape 4 — Fine-tuning NER spaCy (version améliorée)
=====================================================
Améliorations :
  - Hyperparamètres optimisés pour NER militaire (dropout, batch, patience)
  - Config générée proprement sans heuristiques fragiles
  - Evaluation par label affichée post-entraînement
  - Score F1 par entité (WEAPON / MIL_UNIT / MIL_ORG) loggué

Usage :
    python step4_train_model.py --train-data train.spacy --dev-data dev.spacy
"""
from __future__ import annotations
import argparse, logging, subprocess, sys, json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("training_pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("Training_Pipeline")

# Config minimal (base) : seules les sections/valeurs qu'on veut FORCER.
# Toutes les autres clés obligatoires (seed, dropout, score_weights, logger,
# initialize.*, ...) sont complétées automatiquement par `spacy init fill-config`,
# ce qui évite d'avoir à les recopier à la main (source d'erreurs E1015 en chaîne).
BASE_CONFIG = """\
[paths]
train = {train}
dev   = {dev}
vectors = null
init_tok2vec = null

[system]
gpu_allocator = null
seed = 42

[nlp]
lang       = "en"
pipeline   = ["tok2vec","ner"]
batch_size = 128

[components]

[components.tok2vec]
source = "en_core_web_sm"

[components.ner]
source = "en_core_web_sm"
replace_listeners = ["model.tok2vec"]

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path     = ${{paths.train}}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path     = ${{paths.dev}}
max_length = 0

[training]
patience       = 1500
max_epochs     = 0
max_steps      = 3000
eval_frequency = 50
dropout        = 0.1
dev_corpus     = "corpora.dev"
train_corpus   = "corpora.train"

[training.optimizer]
@optimizers = "Adam.v1"
beta1     = 0.9
beta2     = 0.999
eps       = 1e-8
L2        = 1e-6
grad_clip = 1.0

[training.optimizer.learn_rate]
@schedules   = "warmup_linear.v1"
warmup_steps = 250
total_steps  = 3000
initial_rate = 5e-4

[training.batcher]
@batchers         = "spacy.batch_by_words.v1"
discard_oversize  = false
tolerance         = 0.2

[training.batcher.size]
@schedules = "compounding.v1"
start    = 100
stop     = 1000
compound = 1.001

[initialize]
vectors = "en_core_web_sm"
"""

def write_config(config_path: Path, train_path: Path, dev_path: Path) -> None:
    """Écrit un config minimal, puis le complète avec `spacy init fill-config`
    pour que toutes les clés obligatoires (seed, score_weights, logger,
    initialize.components, etc.) soient générées correctement par spaCy lui-même."""
    base_path = config_path.with_suffix(".base.cfg")
    content = BASE_CONFIG.format(
        train=str(train_path.absolute()).replace("\\", "/"),
        dev=str(dev_path.absolute()).replace("\\", "/"),
    )
    base_path.write_text(content, encoding="utf-8")

    cmd = [sys.executable, "-m", "spacy", "init", "fill-config",
           str(base_path), str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        raise SystemExit("Impossible de compléter le config.cfg (fill-config a échoué)")

    logger.info(f"✓ Config écrit dans {config_path}")

def run(cmd: list) -> int:
    result = subprocess.run(cmd)
    return result.returncode

def evaluate_per_label(model_path: Path, dev_path: Path) -> None:
    """Affiche le F1 par label après entraînement."""
    try:
        import spacy
        from spacy.tokens import DocBin
        from spacy.scorer import Scorer

        nlp  = spacy.load(model_path)
        db   = DocBin().from_disk(dev_path)
        docs = list(db.get_docs(nlp.vocab))

        examples = []
        from spacy.training import Example
        for gold in docs:
            pred = nlp(gold.text)
            examples.append(Example(pred, gold))

        scores = nlp.evaluate(examples)
        ents   = scores.get("ents_per_type", {})
        logger.info("\n📊 Scores par label :")
        for label in ["WEAPON", "MIL_UNIT", "MIL_ORG"]:
            s = ents.get(label, {})
            logger.info(
                f"  {label:<12} P={s.get('p',0):.3f}  R={s.get('r',0):.3f}  F1={s.get('f',0):.3f}"
            )
        logger.info(f"  {'OVERALL':<12} F1={scores.get('ents_f',0):.3f}")
    except Exception as e:
        logger.warning(f"Évaluation per-label impossible : {e}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("train.spacy"))
    parser.add_argument("--dev-data",   type=Path, default=Path("dev.spacy"))
    parser.add_argument("--output",     type=Path, default=Path("./output"))
    parser.add_argument("--config",     type=Path, default=Path("config.cfg"))
    parser.add_argument("--base-model", default="en_core_web_sm")
    parser.add_argument("--gpu-id",     type=int,  default=-1)
    args = parser.parse_args()

    # Vérifier le modèle de base
    try:
        import spacy; spacy.load(args.base_model)
        logger.info(f"✓ Modèle '{args.base_model}' disponible")
    except OSError:
        logger.info(f"Téléchargement de '{args.base_model}'...")
        if run([sys.executable, "-m", "spacy", "download", args.base_model]) != 0:
            raise SystemExit(1)

    write_config(args.config, args.train_data, args.dev_data)

    args.output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "spacy", "train", str(args.config),
           "--output", str(args.output),
           "--paths.train", str(args.train_data),
           "--paths.dev",   str(args.dev_data)]
    if args.gpu_id >= 0:
        cmd += ["--gpu-id", str(args.gpu_id)]

    logger.info("🚀 Entraînement lancé...")
    if run(cmd) != 0:
        raise SystemExit("Entraînement échoué")

    logger.info(f"✅ Modèle sauvegardé dans {args.output}/model-best")
    evaluate_per_label(args.output / "model-best", args.dev_data)

if __name__ == "__main__":
    main()