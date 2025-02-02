# Text Classification using HuggingFace Transformers

This repository gives some template code to quickly set up, fine-tune and do inference using HuggingFace LLMs. This is still a work in progress. 

## Installation

``` 
pip install uv
uv venv
uv pip install -r pyproject.toml
```

## Usage

In the file config.json you can make changes to the base config found in src/config.py - then simply activate the uv environment and run:

```
python src/classifier/text_classifier.py
```

## To DO:

- Finish model inference
- Implement Validation
- Add Unit Tests & end-to-end testing.
- Support for mlflow (maybe)?


## Changelog

See `CHANGELOG.md` for release history


