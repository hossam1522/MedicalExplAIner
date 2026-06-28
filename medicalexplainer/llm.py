"""
llm - LLM wrapper with dynamic vLLM model support.

Provides a single :class:`Llm` class that talks to a vLLM server through its
OpenAI-compatible HTTP API (``/v1/chat/completions``).  Any model name not
present in :data:`API_MODELS` is treated as a model served by vLLM; the name
must match what the server was launched with (``vllm serve <model>``).

Google API models (Gemini, Gemma) are also supported via explicit
registration in :data:`API_MODELS`.
"""

import logging
import math
import os
import warnings

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

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
# vLLM endpoint resolution
# ------------------------------------------------------------------

_VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _vllm_base_url() -> str:
    """Return the vLLM OpenAI-compatible base URL.

    Honours ``VLLM_BASE_URL`` (e.g. ``http://gpu-server:8000/v1``).  If the
    value omits the ``/v1`` suffix it is appended, so ``http://host:8000`` and
    ``http://host:8000/v1`` are both accepted.
    """
    raw = os.environ.get("VLLM_BASE_URL", "").strip()
    if not raw:
        raw = _VLLM_DEFAULT_BASE_URL
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    raw = raw.rstrip("/")
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


def _vllm_api_key() -> str:
    """API key for the vLLM server (vLLM ignores it unless started with one)."""
    return os.environ.get("VLLM_API_KEY", "EMPTY") or "EMPTY"


# ------------------------------------------------------------------
# Main LLM class
# ------------------------------------------------------------------


class Llm:
    """Unified LLM wrapper for both vLLM and Google API models.

    Any model name not present in :data:`API_MODELS` is treated as a model
    served by a vLLM instance reachable at :func:`_vllm_base_url`.
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
        # think=True → let the model reason (default); False → disable thinking.
        # Only affects vLLM models whose chat template honours enable_thinking
        # (e.g. Qwen3).  Ignored by everything else.
        self.think = think

        if self.is_api_model:
            self._init_api_model(model_name)
        else:
            self._init_vllm_model(model_name)

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

    def _init_vllm_model(self, name: str) -> None:
        base_url = _vllm_base_url()
        self.client = OpenAI(base_url=base_url, api_key=_vllm_api_key())
        self._check_served(name, base_url)
        logger.debug(
            "Initialised vLLM model: %s (think=%s, base_url: %s)",
            name, self.think, base_url,
        )

    def _check_served(self, name: str, base_url: str) -> None:
        """Verify the vLLM server is reachable and serving *name*."""
        try:
            served = [m.id for m in self.client.models.list().data]
        except Exception as exc:
            raise RuntimeError(
                f"vLLM server not reachable at {base_url}. "
                "Start it with 'vllm serve <model>', or set VLLM_BASE_URL "
                f"to the correct address. ({exc})"
            ) from exc
        if served and name not in served:
            logger.warning(
                "Model %r is not in the vLLM served list %s. "
                "Requests will fail unless the server was launched with this "
                "model (or an alias for it).",
                name, served,
            )

    # ---- LLM calling -------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to OpenAI chat message dicts."""
        role_map = {"system": "system", "ai": "assistant"}
        return [
            {"role": role_map.get(m.type, "user"), "content": m.content}
            for m in messages
        ]

    def _extra_body(self) -> dict | None:
        """vLLM-specific request extras.

        Only emitted when thinking is explicitly disabled, so the default path
        (think=True / standard models) sends a vanilla OpenAI request that any
        server accepts.  ``enable_thinking`` is honoured by reasoning chat
        templates (Qwen3) and ignored otherwise.
        """
        if self.think:
            return None
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def call_llm(self, messages: list[BaseMessage], max_tokens: int = -1) -> str:
        """Invoke the LLM and return the text response.

        Args:
            messages: The conversation messages to send.
            max_tokens: Maximum tokens to generate.  ``-1`` means unlimited.
                Pass a positive value to cap short free-text responses
                (e.g. subquestions, answers).
        """
        if self.is_api_model:
            response = self.llm.invoke(messages)
            return response.content

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            temperature=0,
            max_tokens=None if max_tokens < 0 else max_tokens,
            extra_body=self._extra_body(),
        )
        return (resp.choices[0].message.content or "").strip()

    def call_llm_with_logprobs(
        self, messages: list[BaseMessage]
    ) -> tuple[str, dict[str, float]]:
        """Invoke the LLM and return (text, logprobs_dict).

        For vLLM models, logprobs come from the OpenAI-compatible
        ``/v1/chat/completions`` response.  For API models logprobs are not
        available and the dict is empty.

        The logprobs dict maps each token (``"1"``-``"5"``) to its
        log-probability.  Missing tokens get ``-inf``.
        """
        if self.is_api_model:
            text = self.call_llm(messages)
            return text, {}

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            temperature=0,
            max_tokens=None,  # let the digit (or reasoning chain) complete
            logprobs=True,
            top_logprobs=20,  # 20 captures all of 1-5 as alternatives
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()

        # OpenAI logprobs come back as pydantic objects; normalise to dicts so
        # _extract_logprobs is a pure, easily-tested function.
        content = []
        lp = getattr(choice, "logprobs", None)
        for tok in (getattr(lp, "content", None) or []):
            content.append(
                {
                    "token": tok.token,
                    "logprob": tok.logprob,
                    "top_logprobs": [
                        {"token": a.token, "logprob": a.logprob}
                        for a in (tok.top_logprobs or [])
                    ],
                }
            )

        logprobs_dict = self._extract_logprobs(content)
        prob_dict = {
            k: math.exp(v) if v != float("-inf") else 0.0
            for k, v in logprobs_dict.items()
        }
        logger.debug(
            "Model %s logprobs: %s, probabilities: %s",
            self.model, logprobs_dict, prob_dict,
        )
        return text, logprobs_dict

    @staticmethod
    def _extract_logprobs(content: list[dict]) -> dict[str, float]:
        """Extract per-acuity-level logprobs from OpenAI ``logprobs.content``.

        *content* is the list of per-token objects::

            [{"token": "3", "logprob": -0.01, "top_logprobs": [...]}, ...]

        The acuity digit is the *last* token in the list that is a bare digit
        1-5.  For a standard model that emits only the digit this is the single
        token; for a reasoning model that emits a chain ending in the answer it
        is the final answer digit (scanning from the end skips digits that
        appear inside the reasoning text).  Its ``top_logprobs`` supply the
        alternatives for the other acuity levels.
        """
        logprobs_dict: dict[str, float] = {
            str(i): float("-inf") for i in range(1, 6)
        }
        acuity_tokens = set(logprobs_dict.keys())

        if not content:
            return logprobs_dict

        # Last bare-digit token = the answer; fall back to the last token.
        answer = next(
            (t for t in reversed(content) if t.get("token", "").strip() in acuity_tokens),
            content[-1],
        )

        chosen = answer.get("token", "").strip()
        if chosen in acuity_tokens:
            logprobs_dict[chosen] = answer.get("logprob", float("-inf"))

        for alt in answer.get("top_logprobs", []):
            alt_tok = alt.get("token", "").strip()
            if alt_tok in acuity_tokens:
                logprobs_dict[alt_tok] = alt.get("logprob", float("-inf"))

        return logprobs_dict

    # ---- Prompt methods ----------------------------------------------

    # ESI v4 decision algorithm embedded in all prompts.  Defined once here
    # so every method uses identical, authoritative wording.
    _ESI_ALGORITHM = """\
Apply the ESI v4 algorithm in strict order:

STEP 1 — ESI 1 (Immediate): Does the patient require immediate life-saving intervention RIGHT NOW?
  Yes → ESI 1.  Indicators: cardiac/respiratory arrest, apnea, unresponsiveness, no palpable pulse,
  severe respiratory distress, SpO2 < 80%, BP < 70 mmHg systolic with signs of shock,
  active seizure with no recovery, uncontrolled major haemorrhage.

STEP 2 — ESI 2 (Emergent): Is this a high-risk situation, or does the patient have altered mental
  status or severe pain/distress?
  Yes → ESI 2.  Indicators (any ONE suffices):
  • High-risk chief complaint: chest pain (possible ACS), stroke symptoms, severe dyspnoea,
    anaphylaxis, sepsis (suspected), altered mental status / confusion / lethargy / agitation,
    major trauma.
  • Severe pain: pain score ≥ 7/10.
  • Dangerous vital signs (even if not yet ESI 1): HR > 120 or < 50, RR > 24 or < 8,
    SpO2 < 92%, SBP < 90 mmHg, temperature > 40 °C or < 35 °C.

STEP 3 — ESI 3, 4, or 5 (Stable): How many DISTINCT resource categories will this patient need?
  Resources = labs, IV/IM medications, imaging (X-ray, CT, US, MRI), IV fluids, specialist consult,
  simple procedure (sutures, splint, catheter).
  NOT resources = oral medications, prescriptions, phone referrals, simple wound dressings.
  • ≥ 2 resource categories needed → ESI 3
  •   1 resource category needed → ESI 4
  •   0 resources needed         → ESI 5"""

    def predict_acuity(self, context: str) -> tuple[str, dict[str, float]]:
        """Ask the LLM to predict the triage acuity level (1-5).

        Returns:
            (predicted_acuity, logprobs_dict) where predicted_acuity is a
            string "1"-"5" and logprobs_dict maps "1"-"5" to log-probabilities.
        """
        template = """You are a board-certified emergency medicine physician performing triage.

{esi_algorithm}

Patient information:
{{context}}

Following the ESI v4 algorithm above, what is the triage acuity level for this patient?
Respond with ONLY a single digit: 1, 2, 3, 4, or 5.""".format(
            esi_algorithm=self._ESI_ALGORITHM
        )

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
        template = """You are a board-certified emergency medicine physician performing triage using the ESI v4 algorithm.

{esi_algorithm}

Given the patient information below, generate up to 3 focused questions whose answers would help you decide between ESI levels (e.g. is this ESI 1 vs 2? ESI 2 vs 3? How many resources will be needed?).
Target the decision points in the algorithm that are most uncertain given what you already know.

Patient information:
{{context}}

Output ONLY the questions, one per line. No numbering, no explanations.""".format(
            esi_algorithm=self._ESI_ALGORITHM
        )

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

        template = """You are a board-certified emergency medicine physician performing triage.

{esi_algorithm}

Patient information:
{{context}}

Clinical analysis (answers to targeted ESI decision-point questions):
{{qa_pairs}}

Applying the ESI v4 algorithm to the patient information and the clinical analysis above,
what is the triage acuity level?
Respond with ONLY a single digit: 1, 2, 3, 4, or 5.""".format(
            esi_algorithm=self._ESI_ALGORITHM
        )

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context, qa_pairs=qa_pairs)

        text, logprobs = self.call_llm_with_logprobs(messages)
        return text.strip(), logprobs
