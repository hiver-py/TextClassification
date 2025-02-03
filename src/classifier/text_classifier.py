from config import Config
from load_tokenize_data import load_and_tokenize_data
from utils import compute_metrics, load_config_from_json, load_model
from datasets import Dataset, DatasetDict
import transformers
import logging


logging.basicConfig(filename="text_classifier.log", level=logging.WARNING)
logger = logging.getLogger(__name__)


def run(config: Config, model: transformers.AutoModel, tokenized_dataset: DatasetDict | Dataset) -> None:
    logger.info("Device:", config.device)

    # Set up training arguments
    training_args = transformers.TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    logger.info("Training Arguments: ", training_args)

    # Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, config),
    )
    logger.info("Trainer: ", trainer)

    # Train the model
    if config.train_model:
        logger.info("Training model.")
        trainer.train()

    # Evaluate the model
    logger.info("Running evaluation.")
    eval_results = trainer.evaluate()
    return trainer, eval_results


def main(config: Config, model=None):
    tokenized_dataset = load_and_tokenize_data(config)
    if model is None:
        model = load_model(config)
    run(config, model, tokenized_dataset)


if __name__ == "__main__":
    config = load_config_from_json()
    logger.info("Config: ", config)
    main(config)
