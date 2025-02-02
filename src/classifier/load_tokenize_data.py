from datasets import load_dataset, Dataset, DatasetDict
from .config import Config
import os
import logger
import json

logger = logger.get_logger(__name__)


def load_data(config: Config) -> DatasetDict | Dataset:
    try:
        dataset = load_dataset(config.dataset)
        logger.info("Dataset loaded from Hugging Face Hub.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load from Hugging Face Hub: {e}")


def serve_data(config: Config, dataset: Dataset) -> None:
    classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}
    return None

