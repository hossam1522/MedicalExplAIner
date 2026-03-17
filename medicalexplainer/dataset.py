"""
dataset - Load and merge MIMIC-IV-ED CSV files into patient records.

Reads ``edstays.csv``, ``triage.csv``, ``vitalsign.csv`` and ``diagnosis.csv``
from the data directory, joins them by ``stay_id``, aggregates repeated vital-sign
measurements (median), and exposes a list of patient records ready for LLM
evaluation.

Each record contains:
- Patient identifiers (``subject_id``, ``stay_id``)
- Demographics from *edstays* (``gender``, ``race``, ``arrival_transport``, ``disposition``)
- Triage information (vital signs, ``pain``, ``chiefcomplaint``)
- Aggregated ED vital signs (median of numeric columns, most-frequent rhythm)
- Diagnosis ICD codes
- Ground-truth ``acuity`` (1-5)
"""

import logging
from pathlib import Path

import pandas as pd

from medicalexplainer.logger import configure_logger
from medicalexplainer.paths import DATA_DIR, LOG_PATH

logger = logging.getLogger("dataset")

# All variable groups that can be selected via --variables
VARIABLE_GROUPS: dict[str, list[str]] = {
    "edstays": ["gender", "race", "arrival_transport", "disposition"],
    "triage": [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "chiefcomplaint",
    ],
    "vitalsign": [
        "vs_temperature",
        "vs_heartrate",
        "vs_resprate",
        "vs_o2sat",
        "vs_sbp",
        "vs_dbp",
        "vs_rhythm",
        "vs_pain",
    ],
    "diagnosis": ["diagnoses"],
}

# Flat list of every available variable name
ALL_VARIABLES: list[str] = [v for group in VARIABLE_GROUPS.values() for v in group]


class Dataset:
    """Load MIMIC-IV-ED CSV files and build per-stay patient records."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        variables: list[str] | None = None,
    ) -> None:
        """
        Initialise the dataset.

        Args:
            data_dir: Directory containing the CSV files.  Defaults to ``DATA_DIR``.
            variables: Subset of variable names to include.  ``None`` means all.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
        """
        configure_logger(name="dataset", filepath=LOG_PATH)

        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.variables = variables if variables else list(ALL_VARIABLES)
        self._validate_files()

        self.records: list[dict] = self._build_records()
        logger.info(
            "Loaded %d patient records with %d variables",
            len(self.records),
            len(self.variables),
        )

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------

    _REQUIRED_FILES = ["edstays.csv", "triage.csv", "vitalsign.csv", "diagnosis.csv"]

    def _validate_files(self) -> None:
        for fname in self._REQUIRED_FILES:
            path = self.data_dir / fname
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required file not found: {path}.  "
                    "Place the MIMIC-IV-ED CSV files in the data/ directory."
                )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        logger.debug("Loading %s", path)
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # Vital-sign aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_vitalsigns(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple vital-sign rows per stay into one row (median).

        Numeric columns are aggregated with the *median*.
        The ``rhythm`` column (categorical) is aggregated with the *mode* (most
        frequent value).  The ``pain`` column may contain non-numeric free-text
        entries; numeric values are coerced and aggregated with the median,
        non-numeric values are dropped.
        """
        numeric_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
        agg_dict: dict = {col: "median" for col in numeric_cols if col in df.columns}

        # Pain: coerce to numeric, take median
        if "pain" in df.columns:
            df = df.copy()
            df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
            agg_dict["pain"] = "median"

        # Rhythm: most frequent value
        if "rhythm" in df.columns:

            def _mode(s: pd.Series) -> str | float:
                m = s.dropna().mode()
                return m.iloc[0] if len(m) > 0 else pd.NA

            agg_dict["rhythm"] = _mode

        aggregated = df.groupby("stay_id", as_index=False).agg(agg_dict)
        return aggregated

    # ------------------------------------------------------------------
    # Diagnosis aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
        """Group ICD codes per stay into a single semicolon-separated string."""
        grouped = (
            df.sort_values(["stay_id", "seq_num"])
            .groupby("stay_id", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "diagnoses": "; ".join(
                            f"{row.icd_code} (ICD-{row.icd_version})"
                            for row in g.itertuples()
                        )
                    }
                ),
                include_groups=False,
            )
        )
        return grouped

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    def _build_records(self) -> list[dict]:
        """Merge all CSVs and build the final list of patient records."""
        edstays = self._load_csv("edstays.csv")
        triage = self._load_csv("triage.csv")
        vitalsign_raw = self._load_csv("vitalsign.csv")
        diagnosis_raw = self._load_csv("diagnosis.csv")

        # Keep only stays that have a valid acuity value (ground truth)
        triage = triage.dropna(subset=["acuity"])
        triage["acuity"] = triage["acuity"].astype(int)

        valid_stays = set(triage["stay_id"])
        logger.debug("%d stays with valid acuity", len(valid_stays))

        # Aggregate vitalsigns
        vitalsign = self._aggregate_vitalsigns(vitalsign_raw)
        # Prefix vitalsign columns to avoid collision with triage columns
        vs_rename = {
            col: f"vs_{col}"
            for col in vitalsign.columns
            if col not in ("stay_id", "subject_id")
        }
        vitalsign = vitalsign.rename(columns=vs_rename)

        # Aggregate diagnoses
        diagnoses = self._aggregate_diagnoses(diagnosis_raw)

        # Merge everything on stay_id
        merged = edstays.merge(triage, on=["subject_id", "stay_id"], how="inner")
        merged = merged.merge(vitalsign, on="stay_id", how="left")
        merged = merged.merge(diagnoses, on="stay_id", how="left")

        # Only keep stays present in triage (those with acuity)
        merged = merged[merged["stay_id"].isin(valid_stays)]

        logger.debug("Merged dataset has %d rows", len(merged))

        # Build records
        records: list[dict] = []
        for _, row in merged.iterrows():
            record: dict = {
                "subject_id": int(row["subject_id"]),
                "stay_id": int(row["stay_id"]),
                "acuity": int(row["acuity"]),
            }

            # Add selected variables
            for var in self.variables:
                if var in row.index:
                    val = row[var]
                    record[var] = None if pd.isna(val) else val

            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Context building (text prompt for the LLM)
    # ------------------------------------------------------------------

    def build_context(self, record: dict) -> str:
        """Build a natural-language context string from a patient record.

        Only includes the variables that were selected at init time.

        Args:
            record: A single patient record dict from ``self.records``.

        Returns:
            A formatted string describing the patient data.
        """
        lines: list[str] = []

        # Demographics (edstays)
        demo_vars = [v for v in self.variables if v in VARIABLE_GROUPS["edstays"]]
        if demo_vars:
            lines.append("Demographics:")
            for var in demo_vars:
                val = record.get(var)
                if val is not None:
                    label = var.replace("_", " ").title()
                    lines.append(f"  {label}: {val}")

        # Triage vitals
        triage_vars = [v for v in self.variables if v in VARIABLE_GROUPS["triage"]]
        if triage_vars:
            lines.append("Triage information:")
            for var in triage_vars:
                val = record.get(var)
                if val is not None:
                    label = var.replace("_", " ").title()
                    lines.append(f"  {label}: {val}")

        # Aggregated ED vital signs
        vs_vars = [v for v in self.variables if v in VARIABLE_GROUPS["vitalsign"]]
        if vs_vars:
            lines.append("ED vital signs (median of measurements during stay):")
            for var in vs_vars:
                val = record.get(var)
                if val is not None:
                    label = var.replace("vs_", "").replace("_", " ").title()
                    lines.append(f"  {label}: {val}")

        # Diagnoses
        if "diagnoses" in self.variables:
            diag = record.get("diagnoses")
            if diag:
                lines.append(f"Diagnosis codes: {diag}")

        return "\n".join(lines)
