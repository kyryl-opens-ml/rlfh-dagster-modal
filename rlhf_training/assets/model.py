from dagster import Config, asset, MetadataValue, AssetExecutionContext
from huggingface_hub import hf_hub_download
from datasets import Dataset
import modal


class ModelTrainingConfig(Config):
    pretrained_model_id: str = "cognitivecomputations/dolphin-2.1-mistral-7b"
    peft_model_id: str = "doplhin-dpo-1-epoch"
    num_train_epochs: float = 0.9


@asset(compute_kind="modal")
def trained_model(
    context: AssetExecutionContext,
    config: ModelTrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> str:
    run_training_modal_function = modal.Function.lookup("rlfh-dagster-modal", "run_training_modal")
    hub_model_id = run_training_modal_function.remote(
        pretrained_model_id=config.pretrained_model_id,
        rlhf_model_id=config.peft_model_id,
        train_dataset_pandas=train_dataset.to_pandas(),
        eval_dataset_pands=eval_dataset.to_pandas(),
        num_train_epochs=config.num_train_epochs,
    )
    context.add_output_metadata({"model_url": MetadataValue.url(f"https://huggingface.co/{hub_model_id}")})
    return hub_model_id


@asset(compute_kind="python")
def model_card(context: AssetExecutionContext, trained_model: str) -> str:
    model_card_path = hf_hub_download(repo_id=trained_model, filename="README.md")
    with open(model_card_path, "r") as f:
        content = f.read()
    context.add_output_metadata({"content": MetadataValue.md(content)})
    return content


@asset(compute_kind="modal")
def vibe_check(context: AssetExecutionContext, trained_model: str):
    prompts = [
        "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
        "It's Bengay for muscle relief, a combination of methyl salicylate, menthol, and what other active ingredient commonly found in aspirin?",
        "How can i get rid of llamas in my backyard?",
    ]

    run_sample_inference_modal_function = modal.Function.lookup("rlfh-dagster-modal", "run_sample_inference_modal")

    inference_samples = run_sample_inference_modal_function.remote(prompts=prompts, hub_model_id=trained_model)
    context.add_output_metadata(
        {
            "samples": MetadataValue.json(inference_samples),
        }
    )
