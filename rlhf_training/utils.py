from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import json
from trl import DPOTrainer
import base64
from typing import List, Dict


def run_training(
    pretrained_model_id: str,
    rlhf_model_id: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    num_train_epochs: int = 1,
) -> str:
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_id,
        device_map="auto",
        use_cache=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # to prevent errors with FA
    tokenizer.truncation_side = "left"  # to prevent cutting off last generation

    prompt_length = 1024
    max_seq_length = 1512

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=rlhf_model_id,  # directory to save and repository id
        num_train_epochs=num_train_epochs,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        learning_rate=5e-5,  # 10x higher LR than QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",  # use cosine learning rate scheduler
        logging_steps=25,  # log every 25 steps
        save_steps=500,  # when to save checkpoint
        save_total_limit=2,  # limit the total amount of checkpoints
        evaluation_strategy="steps",  # evaluate every 1000 steps
        eval_steps=700,  # when to evaluate
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        push_to_hub=True,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
    )

    dpo_args = {
        "beta": 0.1,  # The beta factor in DPO loss. Higher beta means less divergence
        "loss_type": "sigmoid",  # The loss type for DPO.
    }

    trainer = DPOTrainer(
        model,
        ref_model=None,  # set to none since we use peft
        peft_config=peft_config,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=prompt_length,
        beta=dpo_args["beta"],
        loss_type=dpo_args["loss_type"],
    )

    trainer.model.print_trainable_parameters()
    # start training, the model will be automatically saved to the hub and the output directory
    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_model()

    kwargs = {
        "finetuned_from": pretrained_model_id,
        "language": "en",
    }
    trainer.create_model_card(**kwargs)

    hub_model_id = trainer.hub_model_id
    del trainer
    del model
    torch.cuda.empty_cache()

    return hub_model_id


def run_sample_inference(prompts: List[str], hub_model_id: str) -> List[Dict[str, str]]:
    model = AutoPeftModelForCausalLM.from_pretrained(hub_model_id, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id)

    # load into pipeline
    merged_model = model.merge_and_unload()
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)

    inference_samples = []
    for prompt in prompts:
        outputs = pipe(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        inference_samples.append(
            {
                "prompt": prompt,
                "generated-answer": outputs[0]["generated_text"][len(prompt) :].strip(),
            }
        )
    return inference_samples


def read_jsonl(file_path: str):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
