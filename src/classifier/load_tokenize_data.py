from datasets import load_dataset, Dataset, DatasetDict
from utils import load_tokenizer
from config import Config
import logging


logger = logging.getLogger(__name__)


def load_data(config: Config) -> DatasetDict | Dataset:
    try:
        dataset = load_dataset(config.dataset)
        if config.target in dataset["train"].features:
            config.num_labels = len(set(dataset["train"][config.target]))
        else:
            logger.warning("'label' column not found in dataset. Unable to set num_labels.")
        logger.info(f"Dataset loaded from HuggingFace Hub. Number of labels: {config.num_labels}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load from HuggingFace Hub: {e}")
        raise RuntimeError(f"Error loading dataset '{config.dataset}': {str(e)}") from e


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
    """Splits huggingface datasets that only has a train key."""
    if len(dataset.keys()) == 1 and "train" in dataset.keys():
        train_test = dataset["train"].train_test_split(
            test_size=0.2, seed=config.seed, stratify=dataset["train"][config.target]
        )
        train_valid = train_test["train"].train_test_split(
            test_size=0.125, seed=config.seed, stratify=train_test["train"][config.target]
        )
        split_dataset = DatasetDict(
            {"train": train_valid["train"], "val": train_valid["test"], "test": train_test["test"]}
        )
        return split_dataset
    else:
        return dataset


def load_and_tokenize_data(config: Config):
    dataset = load_data(config)
    tokenized_data = tokenize_data(dataset, config)
    if len(tokenized_data.keys()) == 1:
        tokenized_data = split_data(tokenized_data)
    return tokenized_data
