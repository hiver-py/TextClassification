import pydantic
import pathlib
import torch


class Config(pydantic.BaseModel):
    """Config to train a text classifier"""

    classifier: str | None = pydantic.Field(
        "SmolLM2-135M", description="Name of model used for training and inference."
    )
    tokenizer: str | None = pydantic.Field(None, description="Name of model used to tokenize data.")
    dataset: str | None = pydantic.Field("stanfordnlp/imdb", description="Name of dataset.")
    target: str | int | None = pydantic.Field(1, description="Name of target variable.")
    train_model: bool = pydantic.Field(True, description="If model should be fine-tuned.")
    num_labels: int | None = pydantic.Field(
        None, description="Number of unique labels, if 'None' will be inferred from train set."
    )
    pretrained_model: bool = pydantic.Field(True, description="If pre-trained model should be used.")
    pretrained_tokenizer: bool = pydantic.Field(True, description="If pre-trained tokenizer should be used.")
    device: str = pydantic.Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Whether to use GPU or CPU."
    )
    seed: int = pydantic.Field(10, description="Seed to use for reproducibility.")
    custom_model_config_path: pathlib.Path | None = pydantic.Field(
        None, description="Path to load model with custom config, only used if pretrained_model=False"
    )
    custom_tokenizer_config_path: pathlib.Path | None = pydantic.Field(
        None, description="Path to load tokenizer with custom config, only used if pretrained_tokenizer=False"
    )
    epochs: int = pydantic.Field(3, description="The number of epochs")
    batch_size: int = pydantic.Field(32, description="The batch size")
    eval_steps: int = pydantic.Field(100, description="Number of steps between evaluations")
    optimizer_type: str = pydantic.Field("adamw", description="Type of optimizer to use")
    scheduler_type: str = pydantic.Field("linear", description="Type of learning rate scheduler")
    early_stopping_patience: int = pydantic.Field(
        3, description="Number of evaluations with no improvement after which training will be stopped"
    )
    lora: bool = pydantic.Field(False, description="Low Rank Adaptation")
    rank: int | None = pydantic.Field(64, description="LoRA attention dimension/rank")
    alpha: float | None = pydantic.Field(16, description="LoRA scaling factor for trained weights")
    learning_rate: float = pydantic.Field(1e-4, description="Learning rate for model training")
    max_length: int | None = pydantic.Field(None, description="Maximum sequence length for input tokens")
    warmup_steps: int | None = pydantic.Field(None, description="Number of warmup steps for learning rate scheduler")
    weight_decay: float | None = pydantic.Field(0.01, description="Weight decay for AdamW optimizer")
    gradient_accumulation_steps: int | None = pydantic.Field(
        1, description="Number of steps to accumulate gradients before performing a backward/update pass"
    )
    fp16: bool = pydantic.Field(False, description="Enable mixed precision training")
    output_dir: str = pydantic.Field("./output", description="Directory to save the trained model")
