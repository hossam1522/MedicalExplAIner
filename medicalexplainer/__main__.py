"""
__main__ - CLI entry point for the medicalexplainer package.

Run with::

    python -m medicalexplainer --dataset <path> [--models m1 m2 ...] [--subtasks] [--limit N]
"""

import argparse
import logging
import sys
from pathlib import Path

from medicalexplainer.evaluator import Evaluator
from medicalexplainer.logger import configure_logger

_LOG_PATH = Path(__file__).parent / "data" / "evaluation" / "medicalexplainer.log"


def main() -> None:
    """Parse CLI arguments and run the evaluation pipeline."""
    configure_logger(name="main", filepath=_LOG_PATH)
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser(
        description="Medical Question Answering System - Evaluate LLM models on a medical dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the JSON dataset file (e.g., medicalexplainer/data/test.final.json)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gemini-3.1-pro-preview"],
        help="List of models to evaluate (e.g., gemini-3.1-pro-preview gpt-oss)",
    )

    parser.add_argument(
        "--subtasks",
        action="store_true",
        help="Enable subtasks division (default: False)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate (useful for testing)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        sys.exit(1)

    if dataset_path.suffix.lower() != ".json":
        logger.error("Dataset must be a JSON file, got: %s", dataset_path.suffix)
        sys.exit(1)

    logger.info("Starting evaluation with dataset: %s", args.dataset)
    logger.info("Models to evaluate: %s", args.models)
    logger.info("Subtasks enabled: %s", args.subtasks)

    if args.limit is not None:
        logger.info("Limiting evaluation to %d questions", args.limit)

    evaluator = Evaluator()

    try:
        evaluator.evaluate(
            models_to_evaluate=args.models,
            json_data_path=str(dataset_path),
            use_subtasks=args.subtasks,
            limit=args.limit,
        )
        logger.info("Evaluation completed successfully!")
    except Exception as exc:
        logger.error("Error during evaluation: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
