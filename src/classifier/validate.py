import transformers
import torch


def validate_model(model, config):
    assert isinstance(model, transformers.PreTrainedModel), "Model should be an instance of PreTrainedModel"
    assert model.num_labels == config.num_labels, "Model's num_labels should match the config"

    if config.pretrained_model:
        assert model.name_or_path == config.classifier, "Model's name or path should match the config classifier"

    # Test forward pass
    dummy_input = torch.randint(0, 1000, (1, 10))  # Adjust input size as needed
    try:
        output = model(dummy_input)
        assert output.logits.shape == (1, config.num_labels), "Model output shape is incorrect"
    except Exception as e:
        raise ValueError(f"Model forward pass failed: {str(e)}")


def validate_tokenizer(tokenizer, config):
    assert isinstance(tokenizer, transformers.PreTrainedTokenizer), (
        "Tokenizer should be an instance of PreTrainedTokenizer"
    )

    if config.pretrained_tokenizer:
        expected_name = config.tokenizer if config.tokenizer else config.classifier
        assert tokenizer.name_or_path == expected_name, "Tokenizer's name or path should match the config"

    # Test tokenization
    test_text = "This is a test sentence."
    try:
        tokens = tokenizer(test_text)
        assert isinstance(tokens, dict), "Tokenizer output should be a dictionary"
        assert "input_ids" in tokens, "Tokenizer output should contain 'input_ids'"
        assert "attention_mask" in tokens, "Tokenizer output should contain 'attention_mask'"
    except Exception as e:
        raise ValueError(f"Tokenizer encoding failed: {str(e)}")
