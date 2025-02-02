from .config import Config
from .load_tokenize_data import (
    load_data,
    serve_data,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logger
#https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
#https://huggingface.co/HuggingFaceTB/SmolLM2-135M
#https://huggingface.co/docs/datasets/en/loading
#https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
#https://github.com/hiver-py
#https://huggingface.co/datasets/wykonos/movies
#https://huggingface.co/datasets?modality=modality:text&size_categories=or:%28size_categories:100K%3Cn%3C1M%29&format=format:csv&sort=trending
logger = logger.get_logger()


def load_config_from_json(file_path: str = "config.json") -> Config:
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return Config(**config_data)


def load_model(config: Config) -> AutoModelForSequenceClassification:
    device = "auto"
    AutoModelForSequenceClassification.from_pretrained()


def load_tokenizer(config: Config) -> AutoTokenizer:
    device = "auto"
    if config.pretrained_model:
        return AutoTokenizer.from_pretrained(config.model)
    else:
        return AutoTokenizer(config.model)

def train(config: Config):
    pass


def evaluate(config: Config):
    pass


def main(
        config,
        model=None,
        tokenizer=None,
        **kwargs
):
    config = load_config_from_json()
    dataset = load_data(config)

    if config.train_model:
        logger.info("Training Model.")
        if model is None:
            model = load_model(config)
        if tokenizer is None:
            tokenizer = load_tokenizer(config)
        train(
            config, 
            model=model, 
            tokenizer=tokenizer, 
            data=dataset
        )

    else:
        evaluate(config)

if __name__ == "__main__":
    main()