"""
llm - LLM wrapper classes and model registry.

Provides a base :class:`Llm` class and concrete subclasses for each supported model,
along with the :data:`MODELS` registry that maps model-name strings to their classes.
"""

import logging
import os
import warnings

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from medicalexplainer.logger import configure_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("llm")


class Llm:
    """Base class for all LLM wrappers used in the evaluation pipeline."""

    def __init__(self, use_subtasks: bool = False):
        """
        Initialize the LLM object.

        Args:
            use_subtasks (bool): Whether to use subtasks division or not.
        """
        load_dotenv()
        self.llm = None
        self.model: str | None = None
        self.use_subtasks = use_subtasks
        self.context = None  # Will be set when answering questions

    def call_llm(self, messages: list[BaseMessage]) -> str:
        """
        Call the LLM with the provided messages and return the response.

        Args:
            messages (list[BaseMessage]): The list of messages to process.

        Returns:
            str: The response from the LLM.
        """
        if self.llm is None:
            raise RuntimeError("LLM has not been initialised; set self.llm first.")
        response = self.llm.invoke(messages)
        return response.content

    def get_subquestions(self, question: str) -> list[str]:
        """
        Generate sub-questions from the LLM for a given medical question.

        Args:
            question (str): The medical question to decompose.

        Returns:
            list[str]: A list of sub-questions.
        """
        template = """You are a medical expert that generates multiple sub-questions related to a medical question.
        The sub-questions should help break down complex medical queries into simpler, more specific questions.

        I do not need the answer to the question. The output should only contain the sub-questions.
        Generate maximum 3 sub-questions. Be as clear and specific as possible.
        The sub-questions should not directly answer the input question but help gather information to answer it.

        Input question: {question}"""

        prompt_decomposition = ChatPromptTemplate.from_template(template)
        messages = {"question": question}

        sub_questions_raw = self.call_llm(prompt_decomposition.format_messages(**messages))
        sub_questions = [q.strip() for q in sub_questions_raw.split("\n") if q.strip()]

        logger.debug(
            "Model: %s, Question: %s, Sub-questions generated: %s",
            self.model,
            question,
            sub_questions,
        )
        return sub_questions

    def answer_subquestion(self, question: str, context: str) -> str:
        """
        Answer a sub-question using the LLM with medical context.

        Args:
            question (str): The question to answer.
            context (str): The medical context/patient record.

        Returns:
            str: The answer to the question.
        """
        template = """You are a medical expert that answers questions about patient medical records.
        Use the following patient medical record to answer the question.
        Provide a clear, concise, and accurate answer based ONLY on the information in the medical record.
        DO NOT make assumptions or provide information not present in the record.

        Question: "{question}"

        Patient Medical Record:
        {context}"""

        prompt = ChatPromptTemplate.from_template(template)
        messages = {"context": context, "question": question}

        answer = self.call_llm(prompt.format_messages(**messages))

        logger.debug("Model: %s, Question: %s, Answer: %s", self.model, question, answer)
        return answer

    def format_qa_pairs(self, questions: list[str], answers: list[str]) -> str:
        """
        Format question/answer pairs into a single readable string.

        Args:
            questions (list[str]): The list of questions.
            answers (list[str]): The list of answers.

        Returns:
            str: The formatted string.
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    def get_final_answer(
        self, question: str, subquestions: list[str], answers: list[str]
    ) -> str:
        """
        Synthesize a final answer from sub-questions and their answers.

        Args:
            question (str): The original medical question.
            subquestions (list[str]): The list of sub-questions.
            answers (list[str]): The list of answers to the sub-questions.

        Returns:
            str: The final synthesized answer.
        """
        template = """You are a medical expert. Here is a set of sub-questions and their answers about a patient:

        {context}

        Use these answers to synthesize a comprehensive and accurate answer to the main question: {question}

        Provide a clear and concise answer that integrates all relevant information from the sub-answers.
        Approximately 10-15 words."""

        prompt = ChatPromptTemplate.from_template(template)
        messages = {
            "context": self.format_qa_pairs(subquestions, answers),
            "question": question,
        }

        final_answer = self.call_llm(prompt.format_messages(**messages))

        logger.debug(
            "Model: %s, Question: %s, Final answer: %s",
            self.model,
            question,
            final_answer,
        )
        return final_answer


class GeminiLlm(Llm):
    """LLM wrapper for Google Gemini models."""

    def __init__(self, use_subtasks: bool = False):
        """
        Initialize the Gemini LLM.

        Args:
            use_subtasks (bool): Whether to use subtasks division or not.
        """
        super().__init__(use_subtasks)
        api_key = os.getenv("GOOGLE_API_KEY") or ""
        os.environ["GOOGLE_API_KEY"] = api_key

        self.model = "gemini-3.1-pro-preview"

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        logger.debug("Using Gemini LLM (%s)", self.model)


class Gemma3Llm(Llm):
    """LLM wrapper for Google Gemma 3 models."""

    def __init__(self, use_subtasks: bool = False):
        """
        Initialize the Gemma 3 LLM.

        Args:
            use_subtasks (bool): Whether to use subtasks division or not.
        """
        super().__init__(use_subtasks)
        api_key = os.getenv("GOOGLE_API_KEY") or ""
        os.environ["GOOGLE_API_KEY"] = api_key

        self.model = "gemma-3-27b-it"

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_tokens=None,
            timeout=None,
        )

        logger.debug("Using Gemma 3 LLM (%s)", self.model)


class GptOssLlm(Llm):
    """LLM wrapper for gpt-oss (Ollama-served) models."""

    def __init__(self, use_subtasks: bool = False):
        """
        Initialize the gpt-oss LLM.

        Args:
            use_subtasks (bool): Whether to use subtasks division or not.
        """
        super().__init__(use_subtasks)

        self.model = "gpt-oss"

        self.llm = ChatOllama(
            model=self.model,
            num_ctx=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        logger.debug("Using gpt-oss LLM")


class OpenBioLlm(Llm):
    """LLM wrapper for OpenBioLLM (Ollama-served) models."""

    def __init__(self, use_subtasks: bool = False):
        """
        Initialize the OpenBioLLM.

        Args:
            use_subtasks (bool): Whether to use subtasks division or not.
        """
        super().__init__(use_subtasks)

        self.model = "openbiollm"

        self.llm = ChatOllama(
            model=self.model,
            num_ctx=8096,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        logger.debug("Using OpenBioLLM")


# Maps model-name strings to their LLM wrapper classes.
MODELS: dict[str, type[Llm]] = {
    "gemini-3.1-pro-preview": GeminiLlm,
    "gemma-3-27b": Gemma3Llm,
    "gpt-oss": GptOssLlm,
    "openbiollm": OpenBioLlm,
}
