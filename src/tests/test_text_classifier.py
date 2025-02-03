from faker import Faker
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pytest


@pytest.fixture
def generate_fake_dataset():
    def _generate(num_samples=1000, test_size=0.2, val_size=0.1):
        fake = Faker()
        data = {
            "text": [fake.text(max_nb_chars=200) for _ in range(num_samples)],
            "label": [fake.random_int(min=0, max=1) for _ in range(num_samples)],
        }
        train_val_data, test_data = train_test_split(
            list(zip(data["text"], data["label"])), test_size=test_size, random_state=42
        )
        train_data, val_data = train_test_split(train_val_data, test_size=val_size / (1 - test_size), random_state=42)
        train_dataset = Dataset.from_dict(
            {"text": [item[0] for item in train_data], "label": [item[1] for item in train_data]}
        )
        val_dataset = Dataset.from_dict(
            {"text": [item[0] for item in val_data], "label": [item[1] for item in val_data]}
        )
        test_dataset = Dataset.from_dict(
            {"text": [item[0] for item in test_data], "label": [item[1] for item in test_data]}
        )
        return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

    return _generate


def test_fake_dataset(generate_fake_dataset):
    dataset_split = generate_fake_dataset()
    assert "train" in dataset_split
    assert "validation" in dataset_split
    assert "test" in dataset_split
    assert len(dataset_split["train"]) == 700
    assert len(dataset_split["test"]) == 200
    assert len(dataset_split["validation"]) == 100
