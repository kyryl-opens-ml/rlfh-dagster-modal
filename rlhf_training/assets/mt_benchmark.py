from dagster import Config, asset, MetadataValue, AssetExecutionContext
import subprocess
from rlhf_training.utils import read_jsonl, encode_image


class MTBenchConfig(Config):
    mt_bench_questions_path: str = "data/mt_bench/question.jsonl"
    original_responses_path: str = "data/mt_bench/model_answer/my-sft.jsonl"
    rlhf_responses_path: str = "data/mt_bench/model_answer/my-rlhf.jsonl"

    sft_model_id: str = "my-sft"
    rlhf_model_id: str = "my-rlhf"


@asset(compute_kind="python")
def mt_bench_questions(context: AssetExecutionContext, config: MTBenchConfig):
    _mt_bench_questions = read_jsonl(config.mt_bench_questions_path)

    context.add_output_metadata(
        {
            "mt_bench_questions": MetadataValue.json(_mt_bench_questions),
        }
    )
    return config.mt_bench_questions_path


@asset(compute_kind="python")
def original_responses(context: AssetExecutionContext, config: MTBenchConfig, mt_bench_questions):
    cmd = f"python FastChat/fastchat/llm_judge/gen_model_answer.py  --model-id {config.sft_model_id} --model-path cognitivecomputations/dolphin-2.1-mistral-7b"
    result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
    _original_responses = read_jsonl(config.original_responses_path)

    context.add_output_metadata(
        {
            "original_responses": MetadataValue.json(_original_responses),
            "cli_output": MetadataValue.text(result.stdout),
        }
    )
    return config.original_responses_path


@asset(compute_kind="python")
def rlhf_responses(
    context: AssetExecutionContext,
    config: MTBenchConfig,
    mt_bench_questions,
    trained_model: str,
):
    cmd = f"python FastChat/fastchat/llm_judge/gen_model_answer.py --model-id {config.rlhf_model_id}  --model-path {trained_model}"
    result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
    _rlhf_responses = read_jsonl(config.rlhf_responses_path)

    context.add_output_metadata(
        {
            "rlhf_responses": MetadataValue.json(_rlhf_responses),
            "cli_output": MetadataValue.text(result.stdout),
        }
    )
    return config.rlhf_responses_path


@asset(compute_kind="python")
def judgment_results(
    context: AssetExecutionContext,
    config: MTBenchConfig,
    original_responses,
    rlhf_responses,
):
    cmd = f"python FastChat/fastchat/llm_judge/gen_judgment.py --model-list {config.sft_model_id} {config.rlhf_model_id} --judge-model gpt-4-1106-preview --mode pairwise-all"
    result_gen_judgment = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)

    cmd = f"python FastChat/fastchat/llm_judge/show_result.py --input-file ./data/mt_bench/model_judgment/gpt-4-1106-preview_pair.jsonl --model-list {config.sft_model_id} {config.rlhf_model_id} --judge-model gpt-4-1106-preview --mode pairwise-all"
    result_show_result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)

    image_path = "win_rate_gpt-4-1106-preview.png"
    image_data = encode_image(image_path)
    md_content = f"![img](data:image/png;base64,{image_data})"

    context.add_output_metadata(
        {
            "plot": MetadataValue.md(md_content),
        }
    )
