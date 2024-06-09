from dagster import Config, asset, MetadataValue, AssetExecutionContext
from datasets import load_dataset
from random import randint
from transformers import AutoTokenizer
from typing import Dict
from datasets import Dataset


class DataConfig(Config):
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    train_data_path: str = "train_dataset.json"
    eval_data_path: str = "eval_dataset.json"
    eval_size: float = 0.1
    sample_training: int = 10_000
    model_id: str = "cognitivecomputations/dolphin-2.1-mistral-7b"


@asset(compute_kind="python")
def rlhf_dataset(config: DataConfig) -> Dict[str, str]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    dataset = load_dataset(config.dataset_name, split="train")
    dataset = dataset.shuffle().select(range(config.sample_training))

    def rec_extract_assistant_messages(messages, index=-1):
        """Recursively extract the last assistant messages from the end of the conversation."""
        if messages[index]["role"] == "assistant":
            return [messages[index]]
        else:
            return rec_extract_assistant_messages(messages, index - 1)

    DEFAULT_SYSTEM_MESSAGE = "You are Dolphin, a helpful AI assistant."

    def create_triplets(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
        """Create the triplets (prompt, chosen, rejected)"""
        prompt_messages = example["chosen"][:-1]
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": default_system_message})

        chosen_messages = rec_extract_assistant_messages(example["chosen"])
        rejected_messages = rec_extract_assistant_messages(example["rejected"])

        return {
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
            "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        }

    dataset = dataset.map(
        create_triplets,
        remove_columns=dataset.features,
        fn_kwargs={"tokenizer": tokenizer},
    )
    dataset = dataset.train_test_split(test_size=config.eval_size)

    dataset["train"].to_json(config.train_data_path, orient="records")
    dataset["test"].to_json(config.eval_data_path, orient="records")

    return {
        "train_path": config.train_data_path,
        "test_path": config.eval_data_path,
    }


@asset(compute_kind="python")
def train_dataset(context: AssetExecutionContext, rlhf_dataset: Dict[str, str]) -> Dataset:
    dataset = load_dataset("json", data_files=rlhf_dataset["train_path"], split="train")
    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset)),
            "sample": MetadataValue.json(dataset[randint(0, len(dataset))]),
        }
    )
    return dataset


@asset(compute_kind="python")
def eval_dataset(context: AssetExecutionContext, rlhf_dataset: Dict[str, str]) -> Dataset:
    dataset = load_dataset("json", data_files=rlhf_dataset["test_path"], split="train")
    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset)),
            "sample": MetadataValue.json(dataset[randint(0, len(dataset))]),
        }
    )
    return dataset
