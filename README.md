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

## Pre-commits

How to run the pre-commits using uv:

```
uv run pre-commit install
uv run pre-commit run -a
```

## To Do:

- Add Tests


## Changelog

See `CHANGELOG.md` for release history


