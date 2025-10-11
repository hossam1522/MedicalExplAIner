import argparse
import sys
import logging
from pathlib import Path
from medicalexplainer.logger import configure_logger
from medicalexplainer.evaluator import Evaluator

configure_logger(name="main", filepath=Path(__file__).parent / "data/evaluation/medicalexplainer.log")
logger = logging.getLogger("main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Question Answering System - Evaluate LLM models on medical dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the JSON dataset file (e.g., medicalexplainer/data/test.final.json)"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gemini-2.0-flash"],
        help="List of models to evaluate (e.g., gemini-2.0-flash qwen2.5-7b llama3.1-8b)"
    )

    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable tools usage (not recommended for medical context)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate (for testing purposes)"
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)

    if not dataset_path.suffix == ".json":
        logger.error(f"Dataset must be a JSON file, got: {dataset_path.suffix}")
        sys.exit(1)

    logger.info(f"Starting evaluation with dataset: {args.dataset}")
    logger.info(f"Models to evaluate: {args.models}")
    logger.info(f"Tools enabled: {args.tools}")

    if args.limit:
        logger.info(f"Limiting evaluation to {args.limit} questions")

    evaluator = Evaluator()

    try:
        evaluator.evaluate(
            models_to_evaluate=args.models,
            json_data_path=str(dataset_path),
            tools=args.tools
        )
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)
