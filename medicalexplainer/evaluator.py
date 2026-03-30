"""
evaluator - Orchestrate the LLM triage acuity prediction pipeline.

The :class:`Evaluator` iterates over patient records, asks each model to
predict the ESI triage acuity (1-5), collects logprobs, and writes
structured results to a CSV file.

No judge LLM is needed: predictions are compared directly against the
ground-truth ``acuity`` column from the *triage* table.
"""

import csv
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from medicalexplainer.dataset import Dataset
from medicalexplainer.llm import Llm
from medicalexplainer.logger import configure_logger
from medicalexplainer.paths import LOG_PATH, RESULTS_DIR

logger = logging.getLogger("evaluator")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MAX_RETRIES = 10
RETRY_BASE_SLEEP = 5  # seconds; exponential backoff multiplier
API_SLEEP = 2.5  # seconds between API calls (rate-limit guard)

_console = Console(stderr=False)


def _make_progress() -> Progress:
    """Build a rich Progress instance with all desired columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[model]}", justify="right"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=_console,
        expand=True,
    )


class Evaluator:
    """Evaluate LLM models on ESI triage acuity prediction."""

    def __init__(self) -> None:
        configure_logger(name="evaluator", filepath=LOG_PATH)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        models: list[str],
        dataset: Dataset,
        *,
        use_subtasks: bool = False,
        limit: int | None = None,
    ) -> Path:
        """Run the evaluation pipeline and write results to a CSV.

        Args:
            models: Model names to evaluate (Ollama or API).
            dataset: A loaded :class:`Dataset` instance.
            use_subtasks: Whether to decompose into sub-questions first.
            limit: Cap the number of patient records to evaluate.

        Returns:
            Path to the generated results CSV file.
        """
        records = dataset.records
        if limit is not None:
            records = records[:limit]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mode_tag = "subtasks" if use_subtasks else "direct"
        output_path = RESULTS_DIR / f"results_{mode_tag}_{timestamp}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "model",
            "subject_id",
            "stay_id",
            "ground_truth_acuity",
            "predicted_acuity",
            "correct",
            "use_subtasks",
            "subquestions",
            "subanswers",
            "logprobs_1",
            "logprobs_2",
            "logprobs_3",
            "logprobs_4",
            "logprobs_5",
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "prob_5",
        ]

        n_records = len(records)
        n_models = len(models)
        total_rows = n_records * n_models
        logger.info(
            "Starting evaluation: %d models x %d records = %d predictions",
            n_models,
            n_records,
            total_rows,
        )

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with _make_progress() as progress:
                # One overall task spanning every model+record combination
                overall_task = progress.add_task(
                    "overall",
                    total=total_rows,
                    model="initialising...",
                )

                for model_idx, model_name in enumerate(models):
                    logger.info("Evaluating model: %s", model_name)

                    # Update spinner label to show current model
                    progress.update(overall_task, model=model_name)

                    llm = self._init_model_with_retry(model_name, use_subtasks)
                    if llm is None:
                        logger.error(
                            "Could not initialise model %s, skipping.", model_name
                        )
                        # Advance progress by all skipped records
                        progress.advance(overall_task, n_records)
                        continue

                    for idx, record in enumerate(records):
                        context = dataset.build_context(record)

                        # Describe what is happening in the progress bar description
                        progress.update(
                            overall_task,
                            model=(
                                f"{model_name}  "
                                f"[dim](model {model_idx + 1}/{n_models})[/dim]"
                            ),
                        )

                        row = self._evaluate_single(
                            llm=llm,
                            model_name=model_name,
                            record=record,
                            context=context,
                            use_subtasks=use_subtasks,
                            idx=idx,
                            total=n_records,
                        )
                        writer.writerow(row)
                        csvfile.flush()

                        progress.advance(overall_task, 1)

        logger.info("Results written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_model_with_retry(
        model_name: str, use_subtasks: bool
    ) -> Llm | None:
        """Try to initialise a model, retrying on transient failures."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return Llm(model_name, use_subtasks=use_subtasks)
            except Exception as exc:
                logger.warning(
                    "Model init attempt %d/%d for '%s' failed: %s",
                    attempt,
                    MAX_RETRIES,
                    model_name,
                    exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_SLEEP * attempt)
        return None

    def _evaluate_single(
        self,
        llm: Llm,
        model_name: str,
        record: dict,
        context: str,
        use_subtasks: bool,
        idx: int,
        total: int,
    ) -> dict:
        """Evaluate a single record with retries and return a result row."""
        stay_id = record["stay_id"]
        subject_id = record["subject_id"]
        ground_truth = record["acuity"]

        logger.info(
            "[%s] Record %d/%d (stay_id=%d)",
            model_name,
            idx + 1,
            total,
            stay_id,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if llm.is_api_model:
                    time.sleep(API_SLEEP)

                if use_subtasks:
                    predicted, logprobs, subquestions, subanswers = (
                        self._predict_with_subtasks(llm, context)
                    )
                else:
                    predicted, logprobs = llm.predict_acuity(context)
                    subquestions = []
                    subanswers = []

                # Validate prediction
                predicted_clean = self._parse_acuity(predicted)

                # Build logprob / probability columns
                prob_dict = {}
                for k in ("1", "2", "3", "4", "5"):
                    lp = logprobs.get(k, float("-inf"))
                    prob_dict[k] = math.exp(lp) if lp != float("-inf") else 0.0

                row = {
                    "model": model_name,
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "ground_truth_acuity": ground_truth,
                    "predicted_acuity": predicted_clean,
                    "correct": predicted_clean == ground_truth,
                    "use_subtasks": use_subtasks,
                    "subquestions": json.dumps(subquestions) if subquestions else "",
                    "subanswers": json.dumps(subanswers) if subanswers else "",
                    "logprobs_1": logprobs.get("1", ""),
                    "logprobs_2": logprobs.get("2", ""),
                    "logprobs_3": logprobs.get("3", ""),
                    "logprobs_4": logprobs.get("4", ""),
                    "logprobs_5": logprobs.get("5", ""),
                    "prob_1": prob_dict.get("1", ""),
                    "prob_2": prob_dict.get("2", ""),
                    "prob_3": prob_dict.get("3", ""),
                    "prob_4": prob_dict.get("4", ""),
                    "prob_5": prob_dict.get("5", ""),
                }

                correct_sym = "[green]OK[/green]" if predicted_clean == ground_truth else "[red]NO[/red]"
                logger.info(
                    "[%s] stay_id=%d  predicted=%s  ground_truth=%d  correct=%s",
                    model_name,
                    stay_id,
                    predicted_clean,
                    ground_truth,
                    predicted_clean == ground_truth,
                )
                return row

            except Exception as exc:
                sleep_time = RETRY_BASE_SLEEP * attempt
                logger.warning(
                    "[%s] Attempt %d/%d failed for stay_id=%d: %s. "
                    "Sleeping %ds...",
                    model_name,
                    attempt,
                    MAX_RETRIES,
                    stay_id,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)

        # All retries exhausted
        logger.error(
            "[%s] All %d retries exhausted for stay_id=%d",
            model_name,
            MAX_RETRIES,
            stay_id,
        )
        return {
            "model": model_name,
            "subject_id": subject_id,
            "stay_id": stay_id,
            "ground_truth_acuity": ground_truth,
            "predicted_acuity": "",
            "correct": False,
            "use_subtasks": use_subtasks,
            "subquestions": "",
            "subanswers": "",
            "logprobs_1": "",
            "logprobs_2": "",
            "logprobs_3": "",
            "logprobs_4": "",
            "logprobs_5": "",
            "prob_1": "",
            "prob_2": "",
            "prob_3": "",
            "prob_4": "",
            "prob_5": "",
        }

    def _predict_with_subtasks(
        self, llm: Llm, context: str
    ) -> tuple[str, dict[str, float], list[str], list[str]]:
        """Generate sub-questions, answer them, then predict acuity.

        Returns:
            (predicted_acuity, logprobs, subquestions, subanswers)
        """
        if llm.is_api_model:
            time.sleep(API_SLEEP)
        subquestions = llm.get_subquestions(context)

        subanswers: list[str] = []
        for sq in subquestions:
            if llm.is_api_model:
                time.sleep(API_SLEEP)
            subanswers.append(llm.answer_subquestion(sq, context))

        if llm.is_api_model:
            time.sleep(API_SLEEP)
        predicted, logprobs = llm.predict_acuity_with_subanswers(
            context, subquestions, subanswers
        )
        return predicted, logprobs, subquestions, subanswers

    @staticmethod
    def _parse_acuity(raw: str) -> int | str:
        """Extract an integer 1-5 from the model's raw response.

        Returns the integer if valid, otherwise the raw string for debugging.
        """
        cleaned = raw.strip()
        # Take only the first character/digit
        for ch in cleaned:
            if ch.isdigit():
                val = int(ch)
                if 1 <= val <= 5:
                    return val
        # Could not parse
        logger.warning("Could not parse acuity from response: %r", raw)
        return cleaned
