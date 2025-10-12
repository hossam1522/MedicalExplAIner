import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
from medicalexplainer.logger import configure_logger
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore", category=DeprecationWarning)
configure_logger(name="llm", filepath=Path(__file__).parent / "data/evaluation/medicalexplainer.log")
logger = logging.getLogger("llm")


class LLM:
    def __init__(self, tools: bool = False):
        """
        Initialize the LLM object

        Args:
            tools (bool): Whether to use tools or not (not used in medical context)
        """
        load_dotenv()
        self.llm = None
        self.model = None
        self.tools = tools
        self.context = None  # Will be set when answering questions

    def call_llm(self, messages: list[BaseMessage]) -> str:
        """
        Call the LLM with the provided messages and return the response.

        Args:
            messages (list[BaseMessage]): The list of messages to process

        Returns:
            str: The response from the LLM
        """
        response = self.llm.invoke(messages)
        return response.content

    def get_subquestions(self, question: str) -> list:
        """
        Get sub-questions from the LLM for medical questions

        Args:
            question (str): The medical question to process

        Returns:
            list: A list of sub-questions
        """
        template = """You are a medical expert that generates multiple sub-questions related to a medical question.
        The sub-questions should help break down complex medical queries into simpler, more specific questions.

        I do not need the answer to the question. The output should only contain the sub-questions.
        Generate maximum 3 sub-questions. Be as clear and specific as possible.
        The sub-questions should not directly answer the input question but help gather information to answer it.

        Input question: {question}"""

        prompt_decomposition = ChatPromptTemplate.from_template(template)
        messages = {"question": question}

        sub_questions = self.call_llm(prompt_decomposition.format_messages(**messages))
        sub_questions = [q.strip() for q in sub_questions.split('\n') if q.strip()]

        logger.debug(f"Model: {self.model}, Question: {question}, Sub-questions generated: {sub_questions}")
        return sub_questions

    def answer_subquestion(self, question: str, context: str) -> str:
        """
        Answer a sub-question using the LLM with medical context

        Args:
            question (str): The question to answer
            context (str): The medical context/patient record

        Returns:
            str: The answer to the question
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

        logger.debug(f"Model: {self.model}, Question: {question}, Answer: {answer}")
        return answer

    def format_qa_pairs(self, questions: list, answers: list) -> str:
        """
        Format the questions and answers into a string
        Args:
            questions (list): The list of questions
            answers (list): The list of answers
        Returns:
            str: The formatted string
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    def get_final_answer(self, question: str, subquestions: list, answers: list) -> str:
        """
        Combine the sub-questions and answers to get a final answer

        Args:
            question (str): The original medical question
            subquestions (list): The list of sub-questions
            answers (list): The list of answers to the sub-questions

        Returns:
            str: The final synthesized answer
        """
        template = """You are a medical expert. Here is a set of sub-questions and their answers about a patient:

        {context}

        Use these answers to synthesize a comprehensive and accurate answer to the main question: {question}

        Provide a clear and concise answer that integrates all relevant information from the sub-answers.
        Approximately 10-15 words."""

        prompt = ChatPromptTemplate.from_template(template)
        messages = {"context": self.format_qa_pairs(subquestions, answers), "question": question}

        final_answer = self.call_llm(prompt.format_messages(**messages))

        logger.debug(f"Model: {self.model}, Question: {question}, Final answer: {final_answer}")
        return final_answer


class LLM_GEMINI(LLM):
    """
    Class for Google Gemini LLM
    """
    def __init__(self, tools: bool = False):
        """
        Initialize the Gemini LLM

        Args:
            tools (bool): Whether to use tools or not (not used in medical context)
        """
        super().__init__(tools)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.model = "gemini-2.0-flash"
        self.tools = tools

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        self.llm = llm
        logger.debug("Using Gemini 2.0 Flash LLM")

class LLM_QWEN_2_5_7B(LLM):
    """
    Class for Qwen2.5 7B LLM
    """
    def __init__(self, tools: bool = False):
        """
        Initialize the Qwen2.5 7B LLM

        Args:
            tools (bool): Whether to use tools or not (not used in medical context)
        """
        super().__init__(tools)

        self.model = "qwen2.5:7b-instruct-fp16"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
            num_ctx=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        self.llm = llm
        logger.debug("Using Qwen2.5 7B LLM")

class LLM_GEMMA_3(LLM):
    """
    Class for Google Gemma 3 LLM
    """
    def __init__(self, tools: bool = False):
        """
        Initialize the Gemma 3 LLM

        Args:
            tools (bool): Whether to use tools or not (not used in medical context)
        """
        super().__init__(tools)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.model = "gemma-3-27b-it"
        self.tools = tools

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_tokens=None,
            timeout=None,
        )

        self.llm = llm
        logger.debug("Using Gemma 3 LLM")

class LLM_LLAMA3_1_8B(LLM):
    """
    Class for Llama3.1 8B LLM
    """
    def __init__(self, tools: bool = False):
        """
        Initialize the Llama3.1 8B LLM

        Args:
            tools (bool): Whether to use tools or not (not used in medical context)
        """
        super().__init__(tools)

        #self.model = "llama3.1:8b-instruct-fp16"
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.tools = tools

        #llm = ChatOllama(
        #    model=self.model,
        #    num_ctx=100000,
        #    temperature=0.7,
        #    top_p=0.8,
        #    top_k=20,
        #)
        llm = VLLM(
            model=self.model,
            trust_remote_code=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            #max_new_tokens=131072,
        )

        self.llm = llm
        logger.debug("Using Llama3.1 8B LLM")

"""
This dictionary maps model names to their respective LLM classes and
if windows context size is small or big.
"""
models = {
    "gemini-2.0-flash": LLM_GEMINI,
    "qwen2.5-7b": LLM_QWEN_2_5_7B,
    "gemma-3-27b": LLM_GEMMA_3,
    "llama3.1-8b": LLM_LLAMA3_1_8B,
}
