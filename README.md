# slm-selective-grounding

Research skeleton for selective grounding with small language models.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv sync --dev
pre-commit install
```

## Sample100 Quickstart

The sample contexts JSONL is gitignored; place it at `artifacts/contexts/asqa_train_k6_sample100.jsonl`.

```bash
python -m scripts.10_run_phases phase1 --contexts_jsonl artifacts/contexts/asqa_train_k6_sample100.jsonl --tag sample100
python -m scripts.10_run_phases phase2 --tag sample100
python -m scripts.10_run_phases phase3 --contexts_jsonl artifacts/contexts/asqa_train_k6_sample100.jsonl --tag sample100
python -m scripts.10_run_phases phase4 --contexts_jsonl artifacts/contexts/asqa_train_k6_sample100.jsonl --tag sample100
python -m scripts.10_run_phases phase5 --contexts_jsonl artifacts/contexts/asqa_train_k6_sample100.jsonl --tag sample100
python -m scripts.10_run_phases report --tag sample100
```

Key outputs:
- `artifacts/phase3_claim_scores_sample100.jsonl`
- `artifacts/phase4_claim_scores_refusal_sample100.jsonl`
- `artifacts/phase5_claim_scores_sample100.jsonl`
- `artifacts/report_sample100.jsonl`

Phase5 eliminates conflicts by claim-level filtering.

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
