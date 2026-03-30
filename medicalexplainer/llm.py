"""
llm - LLM wrapper with dynamic Ollama model support.

Provides a single :class:`Llm` class that works with any Ollama model
without requiring code changes.  Ollama models are automatically pulled
if not already available locally.

Google API models (Gemini, Gemma) are also supported via explicit
registration in :data:`API_MODELS`.
"""

import logging
import math
import os
import subprocess
import warnings

import re

import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from medicalexplainer.logger import configure_logger
from medicalexplainer.paths import LOG_PATH

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("llm")

# ------------------------------------------------------------------
# Google API model definitions (these require API keys)
# ------------------------------------------------------------------

API_MODELS: dict[str, dict] = {
    "gemini-2.5-flash": {
        "backend": "google",
        "model_id": "gemini-2.5-flash-preview-04-17",
        "temperature": 0,
    },
    "gemini-2.0-flash": {
        "backend": "google",
        "model_id": "gemini-2.0-flash",
        "temperature": 0,
    },
    "gemma-3-27b": {
        "backend": "google",
        "model_id": "gemma-3-27b-it",
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
    },
}


# ------------------------------------------------------------------
# Ollama host resolution
# ------------------------------------------------------------------

_OLLAMA_DEFAULT_HOST = "localhost:11434"


def _ollama_base_url() -> str:
    """Return the Ollama base URL, honouring the ``OLLAMA_HOST`` env var.

    Ollama itself reads ``OLLAMA_HOST`` to decide where to listen, so we
    mirror that behaviour: if the variable is set we use it, otherwise we
    fall back to ``localhost:11434``.

    The value may or may not include a scheme; we always normalise to
    ``http://host:port`` (Ollama does not support HTTPS natively).
    """
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        raw = _OLLAMA_DEFAULT_HOST
    # Strip any scheme the user may have included
    for prefix in ("http://", "https://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    # raw is now "host:port" or just "host"
    return f"http://{raw}"


# ------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------


def is_ollama_available() -> bool:
    """Check whether the Ollama service is reachable."""
    try:
        resp = requests.get(f"{_ollama_base_url()}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def ollama_model_exists(model_name: str) -> bool:
    """Check whether *model_name* is already pulled in Ollama."""
    try:
        resp = requests.get(f"{_ollama_base_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for m in models:
                # Ollama reports names like "llama3.1:latest"
                name = m.get("name", "")
                if name == model_name or name.startswith(f"{model_name}:"):
                    return True
        return False
    except requests.ConnectionError:
        return False


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r|\x1b\[[0-9]+[GK]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage returns from *text*."""
    return _ANSI_ESCAPE.sub("", text).strip()


def ollama_pull(model_name: str) -> None:
    """Pull a model into Ollama (blocking)."""
    logger.info("Pulling Ollama model '%s' (this may take a while)...", model_name)
    result = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OLLAMA_HOST": _ollama_base_url().removeprefix("http://")},
    )
    if result.returncode != 0:
        raw_err = result.stderr + result.stdout
        clean_err = _strip_ansi(raw_err)
        # Extract the human-readable error line (starts with "Error:")
        error_line = next(
            (ln for ln in clean_err.splitlines() if ln.startswith("Error:")),
            clean_err,
        )
        raise RuntimeError(
            f"Failed to pull Ollama model '{model_name}': {error_line}"
        )
    logger.info("Successfully pulled model '%s'", model_name)


def ensure_ollama_model(model_name: str) -> None:
    """Make sure *model_name* is available in Ollama, pulling it if necessary."""
    host = _ollama_base_url()
    logger.debug("Using Ollama host: %s", host)
    if not is_ollama_available():
        raise RuntimeError(
            f"Ollama is not reachable at {host}.  "
            "Start it with 'ollama serve' or 'systemctl start ollama', "
            "or set OLLAMA_HOST to the correct address."
        )
    if not ollama_model_exists(model_name):
        ollama_pull(model_name)


# ------------------------------------------------------------------
# Main LLM class
# ------------------------------------------------------------------


class Llm:
    """Unified LLM wrapper for both Ollama and Google API models.

    For Ollama models, instantiation will auto-pull the model if needed.
    Any model name not present in :data:`API_MODELS` is treated as an Ollama
    model.
    """

    def __init__(
        self,
        model_name: str,
        *,
        use_subtasks: bool = False,
        think: bool = True,
    ) -> None:
        configure_logger(name="llm", filepath=LOG_PATH)
        load_dotenv()

        self.model = model_name
        self.use_subtasks = use_subtasks
        self.is_api_model = model_name in API_MODELS
        # think=True → let the model reason (default); False → skip thinking chain.
        # Only has effect on Ollama models that declare the 'thinking' capability.
        self.think = think

        if self.is_api_model:
            self._init_api_model(model_name)
        else:
            self._init_ollama_model(model_name)

    # ---- initialisers ------------------------------------------------

    def _init_api_model(self, name: str) -> None:
        cfg = API_MODELS[name]
        api_key = os.getenv("GOOGLE_API_KEY") or ""
        os.environ["GOOGLE_API_KEY"] = api_key

        kwargs: dict = {
            "model": cfg["model_id"],
            "temperature": cfg.get("temperature", 0),
            "max_tokens": None,
            "timeout": None,
        }
        if "top_p" in cfg:
            kwargs["top_p"] = cfg["top_p"]
        if "top_k" in cfg:
            kwargs["top_k"] = cfg["top_k"]

        self.llm = ChatGoogleGenerativeAI(**kwargs)
        self.is_reasoning_model = False
        logger.debug("Initialised Google API model: %s", name)

    def _init_ollama_model(self, name: str) -> None:
        ensure_ollama_model(name)
        base_url = _ollama_base_url()
        self.llm = ChatOllama(
            model=name,
            base_url=base_url,
            num_ctx=4096,
            temperature=0,
        )
        # Probe once to detect reasoning models (e.g. DeepSeek-R1, gpt-oss).
        # These produce a hidden <think> chain before the visible answer; Ollama
        # surfaces it in message.thinking.  We store the flag so every
        # subsequent call uses the right num_predict without a wasted probe.
        self.is_reasoning_model = self._probe_reasoning(name)
        # If the model supports thinking but the user disabled it, log clearly.
        if self.is_reasoning_model and not self.think:
            logger.info(
                "Model %s: thinking disabled (think=False). "
                "Responses will be faster but less deliberate.",
                name,
            )
        kind = "reasoning" if self.is_reasoning_model else "standard"
        think_tag = f", think={'on' if self.think else 'off'}" if self.is_reasoning_model else ""
        logger.debug(
            "Initialised Ollama model: %s (%s%s, host: %s)",
            name, kind, think_tag, base_url,
        )

    # ---- LLM calling -------------------------------------------------

    def _probe_reasoning(self, model_name: str) -> bool:
        """Detect whether *model_name* is a reasoning (thinking) model.

        Queries ``/api/show`` and checks for ``"thinking"`` in the model's
        declared capabilities list.  This is instantaneous (no inference)
        and reliable: Ollama sets the flag based on the model's architecture,
        not on the content of any particular prompt.
        """
        try:
            resp = requests.post(
                f"{_ollama_base_url()}/api/show",
                json={"model": model_name},
                timeout=10,
            )
            resp.raise_for_status()
            capabilities: list[str] = resp.json().get("capabilities", [])
            is_reasoning = "thinking" in capabilities
            if is_reasoning:
                logger.debug(
                    "Model %s has 'thinking' capability — treating as reasoning model.",
                    model_name,
                )
            return is_reasoning
        except Exception as exc:
            logger.warning(
                "Could not query capabilities for %s: %s. "
                "Assuming standard model.",
                model_name,
                exc,
            )
            return False

    def call_llm(self, messages: list[BaseMessage], max_tokens: int = -1) -> str:
        """Invoke the LLM and return the text response.

        For Ollama models this goes through the raw ``/api/chat`` endpoint so
        that the ``think`` flag (and correct ``num_predict``) are respected.
        For API models it falls back to the LangChain ``invoke`` path.

        Args:
            messages: The conversation messages to send.
            max_tokens: Maximum tokens to generate.  ``-1`` means unlimited
                (Ollama default).  Pass a positive value to cap generation
                for short free-text responses (e.g. subquestions, answers).
        """
        if not self.is_api_model:
            ollama_messages = []
            for m in messages:
                role = "user"
                if m.type == "system":
                    role = "system"
                elif m.type == "ai":
                    role = "assistant"
                ollama_messages.append({"role": role, "content": m.content})
            # The think flag controls whether the reasoning chain is included.
            # max_tokens caps generation for short responses.
            data = self._ollama_post(ollama_messages, num_predict=max_tokens)
            return data.get("message", {}).get("content", "").strip()

        response = self.llm.invoke(messages)
        return response.content

    def call_llm_with_logprobs(
        self, messages: list[BaseMessage]
    ) -> tuple[str, dict[str, float]]:
        """Invoke the LLM and return (text, logprobs_dict).

        For Ollama models, logprobs are obtained via the raw Ollama HTTP API
        (``/api/chat``) because langchain-ollama does not directly expose them.

        For API models, logprobs are not available; the dict will be empty.

        The logprobs dict maps each token (``"1"``-``"5"``) to its
        log-probability.  Missing tokens get ``-inf``.
        """
        if self.is_api_model:
            text = self.call_llm(messages)
            return text, {}

        return self._ollama_chat_with_logprobs(messages)

    def _ollama_chat_with_logprobs(
        self, messages: list[BaseMessage]
    ) -> tuple[str, dict[str, float]]:
        """Call Ollama's ``/api/chat`` endpoint with ``logprobs=True``.

        Uses a single request per call:

        * **Standard models** – ``num_predict=1`` so Ollama emits exactly one
          token (the acuity digit) with its logprobs.
        * **Reasoning models with think=True** – ``num_predict=-1`` (no limit)
          so the thinking chain completes and the visible answer appears in
          ``content``.  Logprobs are extracted from the last token in the list.
        * **Reasoning models with think=False** – Ollama's ``think: false``
          parameter skips the chain entirely; the model answers directly like a
          standard model, so ``num_predict=1`` works again.
        """
        ollama_messages = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            ollama_messages.append({"role": role, "content": m.content})

        # When thinking is disabled the model behaves like a standard model:
        # it emits the answer token directly, so num_predict=1 is sufficient.
        use_reasoning_mode = self.is_reasoning_model and self.think
        num_predict = -1 if use_reasoning_mode else 1
        data = self._ollama_post(ollama_messages, num_predict=num_predict)

        text = data.get("message", {}).get("content", "").strip()
        logprobs_dict = self._extract_logprobs(data, reasoning=use_reasoning_mode)

        prob_dict: dict[str, float] = {
            k: math.exp(lp) if lp != float("-inf") else 0.0
            for k, lp in logprobs_dict.items()
        }
        logger.debug(
            "Model %s logprobs: %s, probabilities: %s",
            self.model,
            logprobs_dict,
            prob_dict,
        )

        return text, logprobs_dict

    def _ollama_post(
        self, ollama_messages: list[dict], *, num_predict: int
    ) -> dict:
        """Send a single request to ``/api/chat`` and return the parsed JSON."""
        payload: dict = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "1h",  # keep model in VRAM for the full evaluation run
            "options": {
                "temperature": 0,
                "num_ctx": 4096,  # patient context is ~500 chars; 4096 is ample
                "num_predict": num_predict,
            },
            "logprobs": True,
            "top_logprobs": 20,  # 20 to capture all digit alternatives 1-5
        }
        # Pass think flag only for reasoning models; ignored by standard models.
        if self.is_reasoning_model:
            payload["think"] = self.think
        resp = requests.post(
            f"{_ollama_base_url()}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_logprobs(data: dict, *, reasoning: bool) -> dict[str, float]:
        """Extract per-acuity-level logprobs from an Ollama response.

        Ollama's ``logprobs`` field is a list of per-token objects::

            [{"token": "3", "logprob": -0.01, "top_logprobs": [...]}, ...]

        For **standard models** (``num_predict=1``) the list has a single entry
        and we read ``top_logprobs`` from it.

        For **reasoning models** the list covers all thinking + content tokens.
        The acuity digit is the *last* token in the list (Ollama appends content
        tokens after thinking tokens).  We read the ``top_logprobs`` from that
        last entry to get alternatives for 1-5.

        ``top_logprobs=20`` is used in the request to ensure that all five
        acuity digits appear as alternatives even when the top choice is very
        confident.
        """
        logprobs_dict: dict[str, float] = {
            str(i): float("-inf") for i in range(1, 6)
        }
        acuity_tokens = set(logprobs_dict.keys())

        token_list: list[dict] = data.get("logprobs", [])

        if not token_list:
            # Legacy Ollama format: top_logprobs is a list of {token: logprob}
            legacy = data.get("top_logprobs", [])
            if legacy:
                entry = legacy[-1] if reasoning else legacy[0]
                for token, lp in entry.items():
                    if token.strip() in acuity_tokens:
                        logprobs_dict[token.strip()] = lp
            return logprobs_dict

        # Find the answer token: for reasoning models it is the last one;
        # for standard models it is the only one.
        answer_obj = token_list[-1] if reasoning else token_list[0]

        # Record the chosen token's logprob
        chosen = answer_obj.get("token", "").strip()
        if chosen in acuity_tokens:
            logprobs_dict[chosen] = answer_obj.get("logprob", float("-inf"))

        # Fill alternatives from top_logprobs
        for alt in answer_obj.get("top_logprobs", []):
            alt_tok = alt.get("token", "").strip()
            if alt_tok in acuity_tokens:
                logprobs_dict[alt_tok] = alt.get("logprob", float("-inf"))

        return logprobs_dict

    # ---- Prompt methods ----------------------------------------------

    def predict_acuity(self, context: str) -> tuple[str, dict[str, float]]:
        """Ask the LLM to predict the triage acuity level (1-5).

        Returns:
            (predicted_acuity, logprobs_dict) where predicted_acuity is a
            string "1"-"5" and logprobs_dict maps "1"-"5" to log-probabilities.
        """
        template = """You are an emergency medicine specialist. Based on the following patient information from an Emergency Department visit, predict the triage acuity level.

The Emergency Severity Index (ESI) triage acuity scale is:
1 = Resuscitation (most severe, life-threatening)
2 = Emergent (high risk, severe pain, or altered mental status)
3 = Urgent (stable but needs multiple resources)
4 = Less urgent (needs one resource)
5 = Non-urgent (no resources needed)

Patient information:
{context}

Based on this information, what is the triage acuity level? Respond with ONLY a single number from 1 to 5."""

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context)

        text, logprobs = self.call_llm_with_logprobs(messages)
        return text.strip(), logprobs

    def get_subquestions(self, context: str) -> list[str]:
        """Generate sub-questions to gather information before predicting acuity.

        Args:
            context: The patient context string.

        Returns:
            A list of up to 3 sub-questions.
        """
        template = """You are an emergency medicine specialist. You need to assess the triage acuity level (1-5) for a patient. Before making your prediction, generate up to 3 specific sub-questions that would help you determine the severity.

The sub-questions should focus on aspects of the patient data that are most relevant for determining acuity.

Patient information:
{context}

Generate up to 3 sub-questions. Output ONLY the sub-questions, one per line."""

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context)

        raw = self.call_llm(messages, max_tokens=128)
        subquestions = [q.strip() for q in raw.split("\n") if q.strip()]
        logger.debug("Generated %d subquestions", len(subquestions))
        return subquestions[:3]

    def answer_subquestion(self, question: str, context: str) -> str:
        """Answer a sub-question given the patient context.

        Args:
            question: The sub-question to answer.
            context: The patient context string.

        Returns:
            The answer string.
        """
        template = """You are an emergency medicine specialist. Answer the following question based ONLY on the patient information provided. Be concise and specific.

Patient information:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context, question=question)
        return self.call_llm(messages, max_tokens=256)

    def predict_acuity_with_subanswers(
        self, context: str, subquestions: list[str], answers: list[str]
    ) -> tuple[str, dict[str, float]]:
        """Predict acuity given sub-question analysis.

        Args:
            context: The patient context string.
            subquestions: List of sub-questions previously generated.
            answers: List of answers to those sub-questions.

        Returns:
            (predicted_acuity, logprobs_dict)
        """
        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(subquestions, answers)
        )

        template = """You are an emergency medicine specialist. Based on the patient information and the analysis below, predict the triage acuity level.

The Emergency Severity Index (ESI) triage acuity scale is:
1 = Resuscitation (most severe, life-threatening)
2 = Emergent (high risk, severe pain, or altered mental status)
3 = Urgent (stable but needs multiple resources)
4 = Less urgent (needs one resource)
5 = Non-urgent (no resources needed)

Patient information:
{context}

Clinical analysis:
{qa_pairs}

Based on all this information, what is the triage acuity level? Respond with ONLY a single number from 1 to 5."""

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context, qa_pairs=qa_pairs)

        text, logprobs = self.call_llm_with_logprobs(messages)
        return text.strip(), logprobs
