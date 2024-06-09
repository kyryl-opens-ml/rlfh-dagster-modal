import modal
from modal import Image
import pandas as pd
import os
from typing import List, Dict

app = modal.App("rlfh-dagster-modal")
env = {"HF_TOKEN": os.getenv("HF_TOKEN")}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/rlfh-dagster-modal:main").env(env)
mount = modal.Mount.from_local_python_packages("rlhf_training", "rlhf_training")
timeout = 6 * 60 * 60


@app.function(image=custom_image, gpu="A100", mounts=[mount], timeout=timeout)
def run_training_modal(
    pretrained_model_id: str,
    rlhf_model_id: str,
    train_dataset_pandas: pd.DataFrame,
    eval_dataset_pands: pd.DataFrame,
    num_train_epochs: float,
):
    from datasets import Dataset
    from rlhf_training.utils import run_training

    model_url = run_training(
        pretrained_model_id=pretrained_model_id,
        rlhf_model_id=rlhf_model_id,
        train_dataset=Dataset.from_pandas(train_dataset_pandas),
        eval_dataset=Dataset.from_pandas(eval_dataset_pands),
        num_train_epochs=num_train_epochs,
    )
    return model_url


@app.function(image=custom_image, gpu="A100", mounts=[mount], timeout=timeout)
def run_sample_inference_modal(prompts: List[str], hub_model_id: str) -> List[Dict[str, str]]:
    from rlhf_training.utils import run_sample_inference

    inference_samples = run_sample_inference(prompts=prompts, hub_model_id=hub_model_id)
    return inference_samples
