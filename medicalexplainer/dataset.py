import os
import json
import logging
from pathlib import Path
from medicalexplainer.logger import configure_logger

configure_logger(name="dataset", filepath=Path(__file__).parent / "data/evaluation/medicalexplainer.log")
logger = logging.getLogger("dataset")


class Dataset:
    def __init__(self, file_path: str):
        """
        Initialize the dataset object with the medical records JSON file provided

        Args:
            file_path (str): The path of the JSON file to process
        """
        if not os.path.exists(file_path):
            logger.error(f'The path {file_path} does not exist')
            raise FileNotFoundError(f'The path {file_path} does not exist')
        elif not os.path.isfile(file_path):
            logger.error(f'The path {file_path} is not a file, please provide a file')
            raise FileExistsError(f'The path {file_path} is not a file, please provide a file')
        elif not file_path.endswith('.json'):
            logger.error(f'The file {file_path} is not a JSON file, please provide a JSON file')
            raise TypeError(f'The file {file_path} is not a JSON file, please provide a JSON file')
        else:
            self.__path = os.path.abspath(file_path)

        # Load medical records data
        self.medical_data = self.__load_data(self.__path)
        self.dataset_items = self.__prepare_dataset_items()

    def __load_data(self, file_path: str) -> dict:
        """
        Load medical records from JSON file
        
        Args:
            file_path (str): The path of the JSON file to load

        Returns:
            dict: The loaded medical data
        """
        logger.debug(f'Loading medical data from {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f'Successfully loaded medical data with {len(data.get("data", []))} records')
        return data

    def __prepare_dataset_items(self) -> list:
        """
        Prepare a structured list of dataset items with context, questions and answers

        Returns:
            list: List of dictionaries containing only context, question, and answer
        """
        logger.debug('Preparing dataset items from medical data')
        dataset_items = []

        for record in self.medical_data.get('data', []):
            for paragraph in record.get('paragraphs', []):
                context = paragraph.get('context', '')

                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])

                    if question and answers:
                        # Create a minimal item with only context, question, and answer
                        item = {
                            'context': context,
                            'question': question,
                            'answer': answers[0].get('text', '')
                        }
                        dataset_items.append(item)

        logger.debug(f'Prepared {len(dataset_items)} dataset items')
        return dataset_items
