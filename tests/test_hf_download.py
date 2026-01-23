from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from slm_selective_grounding.datasets.hf_download import download_datasets


class DummyDataset:
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def select(self, indices) -> "DummyDataset":
        selected = [self._rows[i] for i in indices]
        return DummyDataset(selected)

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets(fake_load_dataset) -> None:
    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    sys.modules["datasets"] = module


def test_hf_download_dry_run(tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_load_dataset(dataset_id, config_name, split=None):
        calls["dataset_id"] = dataset_id
        calls["config_name"] = config_name
        calls["split"] = split
        return DummyDataset(
            [
                {"question": "q1", "long_answer": "a1"},
                {"question": "q2", "long_answer": "a2"},
                {"question": "q3", "long_answer": "a3"},
            ]
        )

    _install_fake_datasets(fake_load_dataset)

    config = {
        "datasets": [
            {
                "id": "din0s/asqa",
                "config_name": None,
                "split": "train",
                "n_examples": 2,
            }
        ],
        "dry_run": True,
    }
    output_manifest = tmp_path / "datasets.json"
    output_root = tmp_path / "raw"

    download_datasets(config, output_manifest, run_id="run123", output_root=output_root)

    expected_path = output_root / "din0s__asqa" / "train.jsonl"
    assert expected_path.exists()
    lines = expected_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["datasets"][0]["path"] == str(expected_path)

    assert calls["dataset_id"] == "din0s/asqa"
    assert calls["config_name"] is None
    assert calls["split"] == "train"


def test_hf_download_preserves_question(tmp_path) -> None:
    def fake_load_dataset(dataset_id, config_name, split=None):
        return DummyDataset(
            [
                {"question": "What is QA?", "long_answer": "Answer here."},
            ]
        )

    _install_fake_datasets(fake_load_dataset)

    config = {
        "datasets": [
            {
                "id": "din0s/asqa",
                "config_name": None,
                "split": "train",
                "n_examples": 1,
            }
        ],
        "dry_run": True,
    }
    output_manifest = tmp_path / "datasets.json"
    output_root = tmp_path / "raw"

    download_datasets(config, output_manifest, run_id="run123", output_root=output_root)

    expected_path = output_root / "din0s__asqa" / "train.jsonl"
    row = json.loads(expected_path.read_text(encoding="utf-8").splitlines()[0])
    assert "question" in row or ("query" in row and row["query"] != "?")
