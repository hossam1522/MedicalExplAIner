"""
paths - Centralised path constants for the medicalexplainer project.

All code that needs to locate files outside the package should import from
here instead of constructing paths manually.

Layout assumed::

    <repo_root>/
    ├── data/
    │   ├── edstays.csv
    │   ├── triage.csv
    │   ├── vitalsign.csv
    │   └── diagnosis.csv
    ├── results/              ← evaluation output (CSV results, logs)
    ├── medicalexplainer/     ← this package
    └── ...
"""

from pathlib import Path

# Repo root = one level up from this file (medicalexplainer/paths.py)
REPO_ROOT: Path = Path(__file__).parent.parent

# Input data
DATA_DIR: Path = REPO_ROOT / "data"

# Output
RESULTS_DIR: Path = REPO_ROOT / "results"
LOG_PATH: Path = RESULTS_DIR / "medicalexplainer.log"
