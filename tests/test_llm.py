"""Tests for the LLM wrappers and MODELS registry."""

import pytest

from medicalexplainer.llm import (
    MODELS,
    GeminiLlm,
    Gemma3Llm,
    GptOssLlm,
    Llm,
    OpenBioLlm,
)


def test_models_registry_has_expected_keys() -> None:
    assert "gemini-3.1-pro-preview" in MODELS
    assert "gemma-3-27b" in MODELS
    assert "gpt-oss" in MODELS
    assert "openbiollm" in MODELS


def test_models_registry_values_are_llm_subclasses() -> None:
    for name, cls in MODELS.items():
        assert issubclass(cls, Llm), f"{name} must be a subclass of Llm"


def test_llm_base_format_qa_pairs() -> None:
    llm = Llm()
    result = llm.format_qa_pairs(["Q1", "Q2"], ["A1", "A2"])
    assert "Question 1: Q1" in result
    assert "Answer 1: A1" in result
    assert "Question 2: Q2" in result
    assert "Answer 2: A2" in result


def test_llm_base_call_llm_raises_without_init() -> None:
    llm = Llm()
    # self.llm is None, should raise RuntimeError
    with pytest.raises(RuntimeError, match="LLM has not been initialised"):
        llm.call_llm([])


def test_gemini_llm_is_subclass() -> None:
    assert issubclass(GeminiLlm, Llm)


def test_gemma3_llm_is_subclass() -> None:
    assert issubclass(Gemma3Llm, Llm)


def test_gpt_oss_llm_is_subclass() -> None:
    assert issubclass(GptOssLlm, Llm)


def test_openbio_llm_is_subclass() -> None:
    assert issubclass(OpenBioLlm, Llm)
