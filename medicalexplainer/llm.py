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
# Ollama helpers
# ------------------------------------------------------------------


def is_ollama_available() -> bool:
    """Check whether the Ollama service is reachable."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def ollama_model_exists(model_name: str) -> bool:
    """Check whether *model_name* is already pulled in Ollama."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
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


def ollama_pull(model_name: str) -> None:
    """Pull a model into Ollama (blocking)."""
    logger.info("Pulling Ollama model '%s' (this may take a while)...", model_name)
    result = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to pull Ollama model '{model_name}': {result.stderr.strip()}"
        )
    logger.info("Successfully pulled model '%s'", model_name)


def ensure_ollama_model(model_name: str) -> None:
    """Make sure *model_name* is available in Ollama, pulling it if necessary."""
    if not is_ollama_available():
        raise RuntimeError(
            "Ollama is not running.  Start it with 'ollama serve' or "
            "'systemctl start ollama'."
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

    def __init__(self, model_name: str, *, use_subtasks: bool = False) -> None:
        configure_logger(name="llm", filepath=LOG_PATH)
        load_dotenv()

        self.model = model_name
        self.use_subtasks = use_subtasks
        self.is_api_model = model_name in API_MODELS

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
        logger.debug("Initialised Google API model: %s", name)

    def _init_ollama_model(self, name: str) -> None:
        ensure_ollama_model(name)
        self.llm = ChatOllama(
            model=name,
            num_ctx=32768,
            temperature=0,
        )
        logger.debug("Initialised Ollama model: %s", name)

    # ---- LLM calling -------------------------------------------------

    def call_llm(self, messages: list[BaseMessage]) -> str:
        """Invoke the LLM and return the text response."""
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
        """Call Ollama's ``/api/chat`` endpoint with ``logprobs=True``."""
        ollama_messages = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            ollama_messages.append({"role": role, "content": m.content})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 32768,
                "num_predict": 1,
            },
            "logprobs": True,
            "top_logprobs": 10,
        }

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "").strip()

        # Extract logprobs for tokens "1" through "5"
        logprobs_dict: dict[str, float] = {
            str(i): float("-inf") for i in range(1, 6)
        }

        # Ollama returns top_logprobs in the response
        top_logprobs = data.get("top_logprobs", [])
        if top_logprobs and len(top_logprobs) > 0:
            first_token_probs = top_logprobs[0]  # dict token -> logprob
            for token, logprob in first_token_probs.items():
                clean = token.strip()
                if clean in logprobs_dict:
                    logprobs_dict[clean] = logprob

        # Also convert to probabilities for logging
        prob_dict: dict[str, float] = {}
        for token, lp in logprobs_dict.items():
            prob_dict[token] = math.exp(lp) if lp != float("-inf") else 0.0

        logger.debug(
            "Model %s logprobs: %s, probabilities: %s",
            self.model,
            logprobs_dict,
            prob_dict,
        )

        return text, logprobs_dict

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

        raw = self.call_llm(messages)
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
        return self.call_llm(messages)

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
