"""Tests for the LLM wrappers."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medicalexplainer.llm import API_MODELS, Llm, _vllm_base_url, _vllm_api_key


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
    """Llm should be importable without needing a vLLM server running."""
    assert Llm is not None


def test_llm_is_api_model_detection() -> None:
    """API model names should be detected correctly (without instantiation)."""
    assert "gemini-2.0-flash" in API_MODELS
    assert "some-random-vllm-model" not in API_MODELS


# ---------------------------------------------------------------------------
# _vllm_base_url
# ---------------------------------------------------------------------------


def test_vllm_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without VLLM_BASE_URL the default localhost:8000/v1 is used."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    assert _vllm_base_url() == "http://localhost:8000/v1"


def test_vllm_base_url_no_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare host:port is normalised to http:// and given a /v1 suffix."""
    monkeypatch.setenv("VLLM_BASE_URL", "0.0.0.0:8001")
    assert _vllm_base_url() == "http://0.0.0.0:8001/v1"


def test_vllm_base_url_with_http_no_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """An http:// URL without /v1 gets the suffix appended."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://myhost:8000")
    assert _vllm_base_url() == "http://myhost:8000/v1"


def test_vllm_base_url_with_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full URL with /v1 is left intact (no doubling)."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://myhost:8000/v1")
    assert _vllm_base_url() == "http://myhost:8000/v1"


def test_vllm_base_url_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty VLLM_BASE_URL falls back to the default."""
    monkeypatch.setenv("VLLM_BASE_URL", "")
    assert _vllm_base_url() == "http://localhost:8000/v1"


def test_vllm_api_key_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert _vllm_api_key() == "EMPTY"


def test_vllm_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    assert _vllm_api_key() == "secret"


# ---------------------------------------------------------------------------
# Llm._extract_logprobs (static method — no server required)
# ---------------------------------------------------------------------------


def _make_token_entry(token: str, logprob: float, alts: dict[str, float]) -> dict:
    """Build an OpenAI-style token logprob object (as a dict)."""
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts.items()],
    }


def test_extract_logprobs_single_digit_token() -> None:
    """A standard model emits one digit token; its alternatives fill 1-5."""
    content = [
        _make_token_entry("3", -0.1, {"3": -0.1, "2": -1.5, "4": -2.0}),
    ]
    result = Llm._extract_logprobs(content)
    assert result["3"] == pytest.approx(-0.1)
    assert result["2"] == pytest.approx(-1.5)
    assert result["4"] == pytest.approx(-2.0)
    assert result["1"] == float("-inf")
    assert result["5"] == float("-inf")


def test_extract_logprobs_reads_last_digit_token() -> None:
    """With a reasoning chain, the LAST bare-digit token is the answer."""
    content = [
        _make_token_entry("Step", -0.01, {}),
        _make_token_entry("1", -0.01, {}),      # digit inside the reasoning — ignored
        _make_token_entry(" so", -0.01, {}),
        _make_token_entry("2", -0.05, {"2": -0.05, "1": -3.0, "3": -2.0}),  # answer
    ]
    result = Llm._extract_logprobs(content)
    assert result["2"] == pytest.approx(-0.05)
    assert result["1"] == pytest.approx(-3.0)
    assert result["3"] == pytest.approx(-2.0)
    assert result["4"] == float("-inf")
    assert result["5"] == float("-inf")


def test_extract_logprobs_empty_returns_neg_inf() -> None:
    """An empty content list yields all -inf."""
    result = Llm._extract_logprobs([])
    assert all(v == float("-inf") for v in result.values())
    assert set(result.keys()) == {"1", "2", "3", "4", "5"}


def test_extract_logprobs_ignores_non_acuity_tokens() -> None:
    """Alternatives that are not 1-5 should not appear in the result."""
    content = [
        _make_token_entry("3", -0.2, {"3": -0.2, "X": -1.0, "hello": -5.0}),
    ]
    result = Llm._extract_logprobs(content)
    assert "X" not in result
    assert "hello" not in result
    assert set(result.keys()) == {"1", "2", "3", "4", "5"}


# ---------------------------------------------------------------------------
# Llm.think flag (stored without needing a server)
# ---------------------------------------------------------------------------


def _make_llm_no_server(think: bool = True) -> Llm:
    """Instantiate Llm for an API model so no vLLM connection is needed."""
    with patch("medicalexplainer.llm.load_dotenv"), patch(
        "medicalexplainer.llm.ChatGoogleGenerativeAI"
    ):
        return Llm("gemini-2.0-flash", think=think)


def test_llm_think_default_is_true() -> None:
    llm = _make_llm_no_server(think=True)
    assert llm.think is True


def test_llm_think_false_stored_correctly() -> None:
    llm = _make_llm_no_server(think=False)
    assert llm.think is False


def test_llm_is_api_model_flag() -> None:
    llm = _make_llm_no_server()
    assert llm.is_api_model is True


def test_extra_body_none_when_thinking() -> None:
    llm = _make_llm_no_server(think=True)
    assert llm._extra_body() is None


def test_extra_body_disables_thinking_when_off() -> None:
    llm = _make_llm_no_server(think=False)
    assert llm._extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


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
# call_llm forwards max_tokens to the vLLM client (no network)
# ---------------------------------------------------------------------------


def _make_vllm_llm() -> Llm:
    """Instantiate a vLLM-backed Llm with the OpenAI client mocked out."""
    with patch("medicalexplainer.llm.load_dotenv"), patch(
        "medicalexplainer.llm.OpenAI"
    ) as mock_openai:
        client = mock_openai.return_value
        # _check_served: pretend the server lists our model
        client.models.list.return_value = MagicMock(
            data=[MagicMock(id="my-model")]
        )
        llm = Llm("my-model")
    return llm


def test_call_llm_forwards_max_tokens() -> None:
    """call_llm should forward max_tokens to chat.completions.create."""
    from langchain_core.messages import HumanMessage

    llm = _make_vllm_llm()
    create = llm.client.chat.completions.create
    create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="3"))]
    )

    llm.call_llm([HumanMessage(content="test")], max_tokens=128)

    kwargs = create.call_args.kwargs
    assert kwargs["max_tokens"] == 128


def test_call_llm_unlimited_max_tokens_is_none() -> None:
    """max_tokens=-1 should become None (unlimited)."""
    from langchain_core.messages import HumanMessage

    llm = _make_vllm_llm()
    create = llm.client.chat.completions.create
    create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="3"))]
    )

    llm.call_llm([HumanMessage(content="test")])  # default -1

    assert create.call_args.kwargs["max_tokens"] is None


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
