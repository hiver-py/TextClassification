import pydantic
import pathlib


class Config(pydantic.BaseModel):
    """Config to train a text classifier"""

    classifier: str | None = pydantic.Field(None, description="Name of model used for training and inference.")
    tokenizer: str | None = pydantic.Field(None, description="Name of model used to tokenize data.")
    dataset: str | None = pydantic.Field(None, description="Name of dataset.")
    target: str | None = pydantic.Field(None, description="Name of target variable.")
    pretrained_model: bool = pydantic.Field(True, description="If pretrained model should be used.")
    pretrained_tokenizer: bool = pydantic.Field(True, description="If pretrained tokenizer should be used.")
    custom_model_config_path: pathlib.Path | None = pydantic.Field(None, "Path to load model with custom config, only used if pretrained_model=False")
    custom_tokenizer_config_path: pathlib.Path | None = pydantic.Field(None, "Path to load tokenizer with custom config, only used if pretrained_tokenizer=False")