# slm-selective-grounding

Research skeleton for selective grounding with small language models.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv sync --dev
pre-commit install
```

## Data flow

```
raw data -> corpus -> indexes -> mixtures -> generations -> preference pairs -> dpo model -> eval report
```

## Pipeline scripts

```bash
python scripts/00_download_datasets.py
python scripts/01_build_corpus.py
python scripts/02_build_retriever_indexes.py
python scripts/03_make_mixtures.py
python scripts/04_generate_baseline_outputs.py
python scripts/05_make_preference_pairs.py
python scripts/06_train_dpo.py
python scripts/07_eval.py
```

## CLI

```bash
python -m slm_selective_grounding --help
```
