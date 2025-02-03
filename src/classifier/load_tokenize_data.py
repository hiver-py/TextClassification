from datasets import load_dataset, Dataset, DatasetDict
from utils import load_tokenizer
from config import Config
import logging


logger = logging.getLogger(__name__)


def load_data(config: Config) -> DatasetDict | Dataset:
    try:
        dataset = load_dataset(config.dataset)
        if "label" in dataset["train"].features:
            config.num_labels = len(set(dataset["train"][config.target]))
        else:
            logger.warning("'label' column not found in dataset. Unable to set num_labels.")
        logger.info(f"Dataset loaded from Hugging Face Hub. Number of labels: {config.num_labels}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load from Hugging Face Hub: {e}")
        raise  # Re-raise the exception after logging


def tokenize_data(dataset: Dataset, config: Config):
    # Load tokenizer
    tokenizer = load_tokenizer(config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def split_data(config, dataset):
    if len(dataset.keys()) == 1:
        train_test = dataset["train"].train_test_split(test_size=0.2, seed=config.seed)
        train_valid = train_test["train"].train_test_split(test_size=0.125, seed=config.seed)
        split_dataset = DatasetDict(
            {"train": train_valid["train"], "validation": train_valid["test"], "test": train_test["test"]}
        )
        return split_dataset


def load_and_tokenize_data(config: Config):
    dataset = load_data(config)
    tokenized_data = tokenize_data(dataset, config)
    if len(tokenized_data.keys()) == 1:
        tokenized_data = split_data(tokenized_data)
    return tokenized_data
