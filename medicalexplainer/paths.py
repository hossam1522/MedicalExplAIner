"""
paths - Centralised path constants for the medicalexplainer project.

All code that needs to locate files outside the package (dataset, logs,
evaluation output) should import from here instead of constructing paths
with ``Path(__file__).parent`` scattered across modules.

Layout assumed::

    <repo_root>/
    ├── data/
    │   ├── test.final.json        ← DATASET_PATH
    │   └── evaluation/            ← LOG_DIR, EVALUATION_DIR
    ├── medicalexplainer/          ← this package
    └── ...
"""

from pathlib import Path

# Repo root = two levels up from this file (medicalexplainer/paths.py)
REPO_ROOT: Path = Path(__file__).parent.parent

# Input data
DATA_DIR: Path = REPO_ROOT / "data"
DATASET_PATH: Path = DATA_DIR / "test.final.json"

# Output / logs
EVALUATION_DIR: Path = DATA_DIR / "evaluation"
LOG_PATH: Path = EVALUATION_DIR / "medicalexplainer.log"
BATCH_REQUESTS_DIR: Path = EVALUATION_DIR / "batch_requests"
BATCH_RESULTS_DIR: Path = EVALUATION_DIR / "batch_results"
