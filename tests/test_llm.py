"""Tests for the LLM wrappers."""

import pytest

from medicalexplainer.llm import API_MODELS, Llm


def test_api_models_registry_has_entries() -> None:
    assert len(API_MODELS) > 0
    assert "gemini-2.5-flash" in API_MODELS or "gemini-2.0-flash" in API_MODELS


def test_api_models_have_backend_key() -> None:
    for name, cfg in API_MODELS.items():
        assert "backend" in cfg, f"API model '{name}' missing 'backend'"
        assert "model_id" in cfg, f"API model '{name}' missing 'model_id'"


def test_llm_class_exists() -> None:
    """Llm should be importable without needing Ollama running."""
    assert Llm is not None


def test_llm_is_api_model_detection() -> None:
    """API model names should be detected correctly (without instantiation)."""
    assert "gemini-2.0-flash" in API_MODELS
    assert "some-random-ollama-model" not in API_MODELS
