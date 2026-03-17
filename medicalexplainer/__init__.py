"""
medicalexplainer - LLM-based triage acuity prediction on MIMIC-IV-ED data.

This package evaluates large language models on their ability to predict
Emergency Severity Index (ESI) triage acuity levels (1-5) from emergency
department patient records.

Public API:
    Dataset    - Load and merge MIMIC-IV-ED CSV files into patient records.
    Evaluator  - Orchestrate the prediction pipeline and write results.
    Llm        - Unified LLM wrapper (Ollama + Google API).
"""

from medicalexplainer.dataset import Dataset
from medicalexplainer.evaluator import Evaluator
from medicalexplainer.llm import Llm

__all__ = ["Dataset", "Evaluator", "Llm"]
