from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import transformers
from config import Config
import json


def compute_metrics(eval_pred: transformers.EvalPrediction, config: Config):
    logits, ground_truth = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "f1-score": f1_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions),
        "recall": recall_score(ground_truth, predictions),
    }


def load_config_from_json(file_path: str = "config.json") -> Config:
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return Config(**config_data)


def load_model(config: Config) -> transformers.AutoModel:
    if config.pretrained_model:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            config.classifier, num_labels=config.num_labels
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        return model
    else:
        model = transformers.AutoModel(config.classifier, num_labels=config.num_labels)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        return model


def load_tokenizer(config: Config) -> transformers.AutoTokenizer:
    tokenizer = config.tokenizer if config.tokenizer else config.classifier
    if config.pretrained_tokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        return tokenizer
    else:
        tokenizer = transformers.AutoTokenizer(tokenizer)
        return tokenizer
