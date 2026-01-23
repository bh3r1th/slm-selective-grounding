from __future__ import annotations


def validate_external_passage(row: dict) -> None:
    required = {
        "doc_id": "doc_id must be a non-empty string",
        "text": "text must be a non-empty string",
        "source": "source must be a non-empty string",
    }
    for key, message in required.items():
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(message)
