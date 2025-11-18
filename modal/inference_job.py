"""Modal inference endpoint for FinSureAI deployments."""

import modal
from modal import Image, Volume, gpu

app = modal.App('finsureai')

image = Image.debian_slim().pip_install_from_requirements('modal/requirements.txt')
volume = Volume.from_name('finsureai-models', create_if_missing=True)

MODEL_CACHE = {}


def _load_model(model_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    MODEL_CACHE[model_path] = (tokenizer, model)
    return tokenizer, model


@app.function(image=image, gpu=gpu.A100(), volumes={'/models': volume}).web_endpoint(method='POST')
def inference_endpoint(body: dict):
    payload = body if isinstance(body, dict) else {}
    prompt = payload.get('prompt') or payload.get('message')
    model_path = payload.get('model_path', '/models/qwen-art')
    max_tokens = int(payload.get('max_tokens', 512))
    temperature = float(payload.get('temperature', 0.2))

    if not prompt:
        return {'error': 'prompt is required'}

    tokenizer, model = _load_model(model_path)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    completion = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return {'output': completion.strip()}
