"""Modal GPU finetuning entrypoint for FinSureAI."""

import os
import pathlib
import tempfile

import modal
from modal import Image, Volume, gpu

app = modal.App('finsureai')

image = (
    Image.debian_slim()
    .pip_install_from_requirements('modal/requirements.txt')
    .run_commands(
        'apt-get update && apt-get install -y git',
        'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
        'pip install git+https://github.com/openpipe/art.git'
    )
    .copy_local_file('modal/reward.py', '/reward.py')
)

volume = Volume.from_name('finsureai-models', create_if_missing=True)

SUPPORTED_MODELS = (
    'Qwen/Qwen3-4B-Instruct-2507',
    'meta-llama/Llama-3.1-70B-Instruct'
)


def _download_dataset(url: str) -> str:
    import requests

    target = pathlib.Path(tempfile.gettempdir()) / f"dataset-{pathlib.Path(url).name}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return str(target)


def _load_dataset(dataset_path: str):
    from datasets import load_dataset

    if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
        return load_dataset('json', data_files={'train': dataset_path})['train']
    if dataset_path.endswith('.zip'):
        return load_dataset('json', data_files={'train': dataset_path})['train']
    # Fallback to public dataset for demos
    return load_dataset('virattt/financial-qa-10K', split='train')


def _format_dataset(dataset, tokenizer):
    def format_example(example):
        messages = [
            {
                'role': 'system',
                'content': 'You are a financial expert. Provide a concise answer based on the context.'
            },
            {'role': 'user', 'content': f"Context: {example['context']}\n\nQuestion: {example['question']}"},
            {'role': 'assistant', 'content': example.get('answer') or example.get('response', '')}
        ]
        return {'text': tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    return dataset.shuffle(seed=42).select(range(min(1000, len(dataset)))).map(format_example)


@app.function(image=image, gpu=gpu.A100(), volumes={'/models': volume}, timeout=60 * 60 * 2)
def finetune_job(dataset_url: str, model_name: str, user_id: str):  # noqa: D401
    """Launch SFT + GRPO finetuning pipeline on Modal GPUs."""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
    from trl import SFTTrainer
    from art import GRPOTrainer
    from reward import compute_reward

    if model_name not in SUPPORTED_MODELS:
        raise ValueError('Unsupported base model')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_path = _download_dataset(dataset_url)
    dataset = _load_dataset(dataset_path)
    dataset = _format_dataset(dataset, tokenizer)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map='auto',
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'v_proj']
    )

    model = prepare_model_for_kbit_training(model)

    args = TrainingArguments(
        output_dir='/models/finetuned',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=200,
        logging_steps=50,
        optim='paged_adamw_8bit',
        weight_decay=0.01,
        warmup_steps=100,
        report_to='none'
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=args,
        max_seq_length=2048
    )

    trainer.train()
    trainer.save_model('/models/finetuned')

    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        '/models/finetuned',
        device_map='auto',
        torch_dtype=torch.float16
    ).merge_and_unload()

    merged_path = '/models/merged'
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    raw_dataset = _load_dataset(dataset_path)

    def prepare_grpo(example):
        messages = [
            {
                'role': 'system',
                'content': 'You are a financial expert. Provide a concise answer based on the context.'
            },
            {'role': 'user', 'content': f"Context: {example['context']}\n\nQuestion: {example['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ground = example.get('answer') or example.get('response', '')
        return {'prompt': prompt, 'ground_truth': ground}

    grpo_dataset = raw_dataset.shuffle(seed=42).select(range(min(1000, len(raw_dataset)))).map(prepare_grpo)
    mapping = {item['prompt']: item['ground_truth'] for item in grpo_dataset}

    def reward_fn(prompts, generations, **_):
        rewards = []
        for prompt, generation in zip(prompts, generations):
            rewards.append(compute_reward(prompt, generation, mapping.get(prompt, '')))
        return rewards

    rl_trainer = GRPOTrainer(
        model=merged_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        rollout_batch_size=8,
        learning_rate=5e-6
    )

    rl_trainer.train(grpo_dataset)
    art_path = '/models/qwen-art'
    os.makedirs(art_path, exist_ok=True)
    rl_trainer.save(art_path)

    inference_function = modal.Function.lookup('finsureai', 'inference_endpoint')
    endpoint_url = getattr(inference_function, 'web_url', 'modal://finsureai/inference_endpoint')

    return {
        'endpoint_url': endpoint_url,
        'artifact_path': art_path,
        'user_id': user_id
    }
