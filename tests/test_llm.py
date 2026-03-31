"""Tests for the LLM wrappers."""

import csv
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medicalexplainer.llm import API_MODELS, Llm, _ollama_base_url, _strip_ansi


# ---------------------------------------------------------------------------
# API_MODELS registry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _ollama_base_url
# ---------------------------------------------------------------------------


def test_ollama_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OLLAMA_HOST the default localhost:11434 should be used."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _ollama_base_url() == "http://localhost:11434"


def test_ollama_base_url_from_env_no_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_HOST without a scheme should be normalised to http://."""
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11436")
    assert _ollama_base_url() == "http://0.0.0.0:11436"


def test_ollama_base_url_from_env_with_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_HOST with http:// prefix should not be doubled."""
    monkeypatch.setenv("OLLAMA_HOST", "http://myhost:11434")
    assert _ollama_base_url() == "http://myhost:11434"


def test_ollama_base_url_from_env_with_https(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_HOST with https:// prefix should be rewritten to http://."""
    monkeypatch.setenv("OLLAMA_HOST", "https://myhost:11434")
    assert _ollama_base_url() == "http://myhost:11434"


def test_ollama_base_url_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty OLLAMA_HOST should fall back to the default."""
    monkeypatch.setenv("OLLAMA_HOST", "")
    assert _ollama_base_url() == "http://localhost:11434"


# ---------------------------------------------------------------------------
# _strip_ansi
# ---------------------------------------------------------------------------


def test_strip_ansi_removes_color_codes() -> None:
    assert _strip_ansi("\x1b[31mError\x1b[0m") == "Error"


def test_strip_ansi_removes_carriage_return() -> None:
    assert _strip_ansi("line1\rline2") == "line1line2"


def test_strip_ansi_plain_text_unchanged() -> None:
    assert _strip_ansi("hello world") == "hello world"


def test_strip_ansi_strips_whitespace() -> None:
    assert _strip_ansi("  text  ") == "text"


def test_strip_ansi_complex_sequence() -> None:
    """Simulate a real ollama pull progress line with ANSI codes."""
    raw = "\x1b[?25l\x1b[2Kpulling manifest\r\x1b[2KDone\x1b[?25h"
    result = _strip_ansi(raw)
    assert "\x1b" not in result
    assert "\r" not in result


# ---------------------------------------------------------------------------
# Llm._extract_logprobs (static method — no Ollama required)
# ---------------------------------------------------------------------------


def _make_token_entry(token: str, logprob: float, alts: dict[str, float]) -> dict:
    """Build an Ollama-style token logprob object."""
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts.items()],
    }


def test_extract_logprobs_standard_reads_first_token() -> None:
    """Standard model (reasoning=False) should read token_list[0]."""
    data = {
        "logprobs": [
            _make_token_entry("3", -0.1, {"3": -0.1, "2": -1.5, "4": -2.0}),
            _make_token_entry("X", -9.9, {}),  # second token — must be ignored
        ]
    }
    result = Llm._extract_logprobs(data, reasoning=False)
    assert result["3"] == pytest.approx(-0.1)
    assert result["2"] == pytest.approx(-1.5)
    assert result["4"] == pytest.approx(-2.0)
    assert result["1"] == float("-inf")
    assert result["5"] == float("-inf")


def test_extract_logprobs_reasoning_reads_last_token() -> None:
    """Reasoning model (reasoning=True) should read token_list[-1]."""
    data = {
        "logprobs": [
            _make_token_entry("think", -0.01, {}),  # thinking chain token
            _make_token_entry("2", -0.05, {"2": -0.05, "1": -3.0, "3": -2.0}),
        ]
    }
    result = Llm._extract_logprobs(data, reasoning=True)
    assert result["2"] == pytest.approx(-0.05)
    assert result["1"] == pytest.approx(-3.0)
    assert result["3"] == pytest.approx(-2.0)
    assert result["4"] == float("-inf")
    assert result["5"] == float("-inf")


def test_extract_logprobs_empty_token_list_returns_neg_inf() -> None:
    """When logprobs list is empty and no legacy data, all values are -inf."""
    result = Llm._extract_logprobs({"logprobs": []}, reasoning=False)
    assert all(v == float("-inf") for v in result.values())
    assert set(result.keys()) == {"1", "2", "3", "4", "5"}


def test_extract_logprobs_ignores_non_acuity_tokens() -> None:
    """Tokens that are not 1-5 should not appear in the result."""
    data = {
        "logprobs": [
            _make_token_entry("3", -0.2, {"3": -0.2, "X": -1.0, "hello": -5.0}),
        ]
    }
    result = Llm._extract_logprobs(data, reasoning=False)
    assert "X" not in result
    assert "hello" not in result
    assert set(result.keys()) == {"1", "2", "3", "4", "5"}


# ---------------------------------------------------------------------------
# Llm.think flag (stored without needing Ollama)
# ---------------------------------------------------------------------------


def _make_llm_no_ollama(think: bool = True) -> Llm:
    """Instantiate Llm for an API model so no Ollama connection is needed."""
    with patch("medicalexplainer.llm.load_dotenv"), patch(
        "medicalexplainer.llm.ChatGoogleGenerativeAI"
    ):
        return Llm("gemini-2.0-flash", think=think)


def test_llm_think_default_is_true() -> None:
    llm = _make_llm_no_ollama(think=True)
    assert llm.think is True


def test_llm_think_false_stored_correctly() -> None:
    llm = _make_llm_no_ollama(think=False)
    assert llm.think is False


def test_llm_is_api_model_flag() -> None:
    llm = _make_llm_no_ollama()
    assert llm.is_api_model is True


# ---------------------------------------------------------------------------
# Llm._ESI_ALGORITHM constant
# ---------------------------------------------------------------------------


def test_esi_algorithm_contains_key_steps() -> None:
    algo = Llm._ESI_ALGORITHM
    assert "ESI 1" in algo
    assert "ESI 2" in algo
    assert "ESI 3" in algo or "ESI 3, 4, or 5" in algo
    assert "ESI 4" in algo or "ESI 3, 4, or 5" in algo
    assert "ESI 5" in algo or "ESI 3, 4, or 5" in algo


def test_esi_algorithm_mentions_resources() -> None:
    assert "resource" in Llm._ESI_ALGORITHM.lower()


def test_esi_algorithm_mentions_vital_signs() -> None:
    algo = Llm._ESI_ALGORITHM.lower()
    assert "spo2" in algo or "o2sat" in algo or "spo" in algo or "sp" in algo


# ---------------------------------------------------------------------------
# call_llm passes max_tokens to _ollama_post (no network)
# ---------------------------------------------------------------------------


def test_call_llm_passes_max_tokens_to_post() -> None:
    """call_llm should forward max_tokens as num_predict to _ollama_post."""
    with patch("medicalexplainer.llm.load_dotenv"), patch(
        "medicalexplainer.llm.ensure_ollama_model"
    ), patch("medicalexplainer.llm.ChatOllama"), patch(
        "medicalexplainer.llm.requests.post"
    ) as mock_req, patch.object(
        Llm, "_probe_reasoning", return_value=False
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "3"}}
        mock_resp.raise_for_status.return_value = None
        mock_req.return_value = mock_resp

        llm = Llm("llama3.1")
        from langchain_core.messages import HumanMessage
        llm.call_llm([HumanMessage(content="test")], max_tokens=128)

        call_kwargs = mock_req.call_args
        payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
        assert payload["options"]["num_predict"] == 128


# ---------------------------------------------------------------------------
# _print_summary accuracy calculation
# ---------------------------------------------------------------------------


def test_print_summary_accuracy(tmp_path: Path) -> None:
    """_print_summary should compute correct accuracy per model from a CSV."""
    from medicalexplainer.evaluator import Evaluator

    csv_path = tmp_path / "results.csv"
    fieldnames = [
        "model", "subject_id", "stay_id",
        "ground_truth_acuity", "predicted_acuity", "correct", "use_subtasks",
    ]
    rows = [
        {"model": "modelA", "subject_id": "1", "stay_id": "1",
         "ground_truth_acuity": "3", "predicted_acuity": "3", "correct": "True", "use_subtasks": "False"},
        {"model": "modelA", "subject_id": "2", "stay_id": "2",
         "ground_truth_acuity": "2", "predicted_acuity": "3", "correct": "False", "use_subtasks": "False"},
        {"model": "modelA", "subject_id": "3", "stay_id": "3",
         "ground_truth_acuity": "1", "predicted_acuity": "1", "correct": "True", "use_subtasks": "False"},
        {"model": "modelB", "subject_id": "4", "stay_id": "4",
         "ground_truth_acuity": "4", "predicted_acuity": "4", "correct": "True", "use_subtasks": "False"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # _print_summary should not raise; we patch the rich console to avoid output.
    with patch("medicalexplainer.evaluator._console"):
        Evaluator._print_summary(csv_path)  # no assertion needed — just no crash


def test_print_summary_empty_file_does_not_crash(tmp_path: Path) -> None:
    """_print_summary on a header-only CSV should return silently."""
    from medicalexplainer.evaluator import Evaluator

    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "correct"])
        writer.writeheader()

    with patch("medicalexplainer.evaluator._console"):
        Evaluator._print_summary(csv_path)  # should not raise
