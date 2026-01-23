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
python scripts/00_download_datasets.py --config configs/datasets.yaml
python scripts/01_build_corpus.py --config configs/datasets.yaml
python scripts/01b_build_qa_samples.py --dataset asqa --split train
python scripts/01b_build_qa_samples.py --dataset alce --split train
python scripts/02_build_retriever_indexes.py
python scripts/02_build_retriever_indexes.py --corpus wiki_leads
python scripts/03_build_external_corpus.py
python scripts/04_mix_contexts.py --dataset <dataset>
python scripts/03_make_mixtures.py
python scripts/04_generate_baseline_outputs.py
python scripts/05_make_preference_pairs.py
python scripts/06_train_dpo.py
python scripts/07_eval.py
```

The external corpus builder expects `data/external/wiki_leads.jsonl` with
`{"doc_id","title","text","source"}` fields and writes
`artifacts/corpus/wiki_leads.jsonl` for BM25 indexing and mixing.

The default dataset config pulls Gemma 3 1B IDs and Hugging Face datasets listed in
`configs/default.yaml`, with `dry_run: true` to download 10 examples per dataset. Set
`dry_run: false` in `configs/datasets.yaml` to download the full splits.

## CLI

```bash
python -m slm_selective_grounding --help
```
