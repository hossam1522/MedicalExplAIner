"""
dataset - Dataset loading and validation for medical QA evaluation.

Provides the :class:`Dataset` class that validates and loads a SQuAD-formatted
medical QA JSON file, then flattens it into a list of
``{context, question, answer}`` dictionaries.
"""

import json
import logging
from pathlib import Path

from medicalexplainer.logger import configure_logger
from medicalexplainer.paths import LOG_PATH

logger = logging.getLogger("dataset")


class Dataset:
    """Load and validate a SQuAD-formatted medical QA dataset."""

    def __init__(self, file_path: str) -> None:
        """
        Initialize the dataset from a JSON file.

        Args:
            file_path (str): Path to the JSON file to process.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            FileExistsError: If *file_path* is not a regular file.
            TypeError: If *file_path* does not have a ``.json`` extension.
        """
        configure_logger(name="dataset", filepath=LOG_PATH)

        path = Path(file_path)

        if not path.exists():
            logger.error("The path %s does not exist", file_path)
            raise FileNotFoundError(f"The path {file_path} does not exist")
        if not path.is_file():
            logger.error("The path %s is not a file, please provide a file", file_path)
            raise FileExistsError(
                f"The path {file_path} is not a file, please provide a file"
            )
        if path.suffix.lower() != ".json":
            logger.error(
                "The file %s is not a JSON file, please provide a JSON file", file_path
            )
            raise TypeError(
                f"The file {file_path} is not a JSON file, please provide a JSON file"
            )

        self._path = path.resolve()

        self.medical_data = self._load_data(self._path)
        self.dataset_items = self._prepare_dataset_items()

    def _load_data(self, file_path: Path) -> dict:
        """
        Load medical records from a JSON file.

        Args:
            file_path (Path): Resolved path to the JSON file.

        Returns:
            dict: The parsed medical data.
        """
        logger.debug("Loading medical data from %s", file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(
            "Successfully loaded medical data with %d records",
            len(data.get("data", [])),
        )
        return data

    def _prepare_dataset_items(self) -> list[dict]:
        """
        Flatten the SQuAD-like structure into a list of QA items.

        Returns:
            list[dict]: Each item contains ``context``, ``question``, and ``answer``.
        """
        logger.debug("Preparing dataset items from medical data")
        dataset_items: list[dict] = []

        for record in self.medical_data.get("data", []):
            for paragraph in record.get("paragraphs", []):
                context = paragraph.get("context", "")

                for qa in paragraph.get("qas", []):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])

                    if question and answers:
                        dataset_items.append(
                            {
                                "context": context,
                                "question": question,
                                "answer": answers[0].get("text", ""),
                            }
                        )

        logger.debug("Prepared %d dataset items", len(dataset_items))
        return dataset_items
