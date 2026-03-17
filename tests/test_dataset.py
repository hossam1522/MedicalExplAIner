"""Tests for the Dataset class."""

import json
import pytest
from pathlib import Path

from medicalexplainer.dataset import Dataset


@pytest.fixture()
def valid_json_file(tmp_path: Path) -> Path:
    """Create a minimal valid SQuAD-style JSON dataset file."""
    data = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": "Patient has hypertension.",
                        "qas": [
                            {
                                "question": "What condition does the patient have?",
                                "answers": [{"text": "hypertension", "answer_start": 12}],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


def test_dataset_loads_valid_file(valid_json_file: Path) -> None:
    dataset = Dataset(str(valid_json_file))
    assert len(dataset.dataset_items) == 1


def test_dataset_item_structure(valid_json_file: Path) -> None:
    dataset = Dataset(str(valid_json_file))
    item = dataset.dataset_items[0]
    assert "context" in item
    assert "question" in item
    assert "answer" in item


def test_dataset_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        Dataset("/nonexistent/path/data.json")


def test_dataset_raises_for_non_json_extension(tmp_path: Path) -> None:
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError):
        Dataset(str(txt_file))


def test_dataset_raises_for_directory(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        Dataset(str(tmp_path))
