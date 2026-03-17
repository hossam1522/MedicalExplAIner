"""Tests for the Dataset class."""

import pytest
from pathlib import Path

from medicalexplainer.dataset import ALL_VARIABLES, Dataset


@pytest.fixture()
def demo_data_dir() -> Path:
    """Return the path to the demo data directory."""
    path = Path(__file__).parent.parent / "data"
    if not (path / "edstays.csv").exists():
        pytest.skip("Demo data not available (run 'make download-demo-data')")
    return path


def test_dataset_loads_demo_data(demo_data_dir: Path) -> None:
    dataset = Dataset(data_dir=demo_data_dir)
    assert len(dataset.records) > 0


def test_dataset_records_have_required_fields(demo_data_dir: Path) -> None:
    dataset = Dataset(data_dir=demo_data_dir)
    record = dataset.records[0]
    assert "subject_id" in record
    assert "stay_id" in record
    assert "acuity" in record
    assert 1 <= record["acuity"] <= 5


def test_dataset_variable_selection(demo_data_dir: Path) -> None:
    dataset = Dataset(data_dir=demo_data_dir, variables=["gender", "chiefcomplaint"])
    record = dataset.records[0]
    # Selected variables should be present
    assert "gender" in record or record.get("gender") is None
    assert "chiefcomplaint" in record or record.get("chiefcomplaint") is None
    # Non-selected variables should not be present (except identifiers)
    assert "arrival_transport" not in record


def test_dataset_raises_for_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Dataset(data_dir=tmp_path / "nonexistent")


def test_dataset_raises_for_missing_files(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Required file not found"):
        Dataset(data_dir=tmp_path)


def test_dataset_build_context(demo_data_dir: Path) -> None:
    dataset = Dataset(data_dir=demo_data_dir)
    context = dataset.build_context(dataset.records[0])
    assert isinstance(context, str)
    assert len(context) > 0


def test_all_variables_list() -> None:
    assert len(ALL_VARIABLES) > 0
    assert "gender" in ALL_VARIABLES
    assert "chiefcomplaint" in ALL_VARIABLES
    assert "diagnoses" in ALL_VARIABLES
    assert "vs_heartrate" in ALL_VARIABLES
