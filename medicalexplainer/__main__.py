"""
__main__ - CLI entry point for the medicalexplainer package.

Run with::

    python -m medicalexplainer --models <model1> [model2 ...] [--subtasks] [--limit N]
                               [--variables var1 var2 ...] [--data-dir DIR]
                               [--no-think]
"""

import argparse
import logging
import sys

from medicalexplainer.dataset import ALL_VARIABLES, Dataset
from medicalexplainer.evaluator import Evaluator
from medicalexplainer.logger import configure_logger
from medicalexplainer.paths import DATA_DIR, LOG_PATH


def main() -> None:
    """Parse CLI arguments and run the evaluation pipeline."""
    configure_logger(name="main", filepath=LOG_PATH)
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser(
        description=(
            "MedicalExplAIner - Predict ESI triage acuity (1-5) using LLMs "
            "on MIMIC-IV-ED data"
        ),
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more model names. Ollama models are auto-pulled if not "
            "present. API models: gemini-2.5-flash, gemini-2.0-flash, gemma-3-27b"
        ),
    )

    parser.add_argument(
        "--subtasks",
        action="store_true",
        help="Decompose into sub-questions before predicting acuity (default: off)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of patient records to evaluate (useful for testing)",
    )

    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Subset of variables to include in the patient context. "
            f"Available: {', '.join(ALL_VARIABLES)}.  Default: all."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Directory containing MIMIC-IV-ED CSV files (default: {DATA_DIR})",
    )

    parser.add_argument(
        "--no-think",
        action="store_true",
        help=(
            "Disable chain-of-thought thinking for reasoning models "
            "(faster, less deliberate; no effect on standard models)"
        ),
    )

    args = parser.parse_args()

    # Validate variables
    if args.variables:
        invalid = [v for v in args.variables if v not in ALL_VARIABLES]
        if invalid:
            logger.error(
                "Unknown variables: %s. Available: %s",
                invalid,
                ALL_VARIABLES,
            )
            sys.exit(1)

    logger.info("Models to evaluate: %s", args.models)
    logger.info("Subtasks enabled: %s", args.subtasks)
    logger.info(
        "Variables: %s", args.variables if args.variables else "all"
    )
    if args.limit is not None:
        logger.info("Limiting evaluation to %d records", args.limit)
    think = not args.no_think
    logger.info("Thinking enabled: %s", think)

    try:
        dataset = Dataset(
            data_dir=args.data_dir,
            variables=args.variables,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Download the demo data with: make download-demo-data"
        )
        sys.exit(1)

    evaluator = Evaluator()

    try:
        output_path = evaluator.evaluate(
            models=args.models,
            dataset=dataset,
            use_subtasks=args.subtasks,
            limit=args.limit,
            think=think,
        )
        logger.info("Evaluation complete! Results: %s", output_path)
    except Exception as exc:
        logger.error("Error during evaluation: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
