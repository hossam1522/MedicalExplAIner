"""
medicalexplainer - LLM-based medical question answering and evaluation framework.

This package provides tools to evaluate large language models on medical QA datasets
by optionally decomposing complex queries into sub-questions before answering.

Public API:
    Dataset            - Load and validate a SQuAD-formatted medical QA dataset.
    Evaluator          - Orchestrate the evaluation loop and generate result charts.
    EvaluatorGptBatch  - Evaluator variant that uses the OpenAI Batch API with logprobs.
"""

from medicalexplainer.dataset import Dataset
from medicalexplainer.evaluator import Evaluator
from medicalexplainer.evaluator_gpt_batch import EvaluatorGptBatch  # noqa: F401

__all__ = ["Dataset", "Evaluator", "EvaluatorGptBatch"]
