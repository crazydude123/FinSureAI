import modal
from modal import Image, App, gpu

# Define the Modal app
app = App("finsure-ai")

# Create a custom image with all dependencies
image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "apt-get update && apt-get install -y git",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install bitsandbytes",  # Install bitsandbytes for CUDA on Modal
        "pip install git+https://github.com/openpipe/art.git"  # Install ART from GitHub
    )
    .copy_local_file("reward.py", "/reward.py")  # Copy reward.py into the image (relative to modal_app.py location)
)

# Volume for storing models
volume = modal.Volume.from_name("finsure-models", create_if_missing=True)

@app.function(
    image=image,
    gpu=gpu.A100(),  # Can be changed to H100 or any supported GPU
    volumes={"/models": volume},
    timeout=7200  # 2 hour timeout (increased for SFT + GRPO training)
)
def finetune_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
    from trl import SFTTrainer

    # Model name
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure GPU is available
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name()}")

    # Load dataset
    dataset = load_dataset("virattt/financial-qa-10K", split="train")

    # Format function for chat template
    def format_example(example):
        messages = [
            {"role": "system", "content": "You are a financial expert. Provide a concise answer to the question based on the given context."},
            {"role": "user", "content": f"Context: {example['context']}\n\nQuestion: {example['question']}"},
            {"role": "assistant", "content": example['answer']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    # Apply formatting (subsample to 1000 for faster testing; remove for full)
    dataset = dataset.shuffle(seed=42).select(range(1000)).map(format_example)

    print(dataset[0]['text'])  # Check one example

    # Quantization config for memory efficiency
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Qwen-specific; adjust if needed
    )

    model = prepare_model_for_kbit_training(model)

    # Training arguments
    args = TrainingArguments(
        output_dir="/models/finetuned_qwen",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        warmup_steps=100,
        evaluation_strategy="no",  # No eval set; add if you split dataset
        report_to="none"  # Disable wandb/etc.
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=args,
        max_seq_length=2048  # Adjust based on your data lengths
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model("/models/finetuned_qwen")

    # Load PEFT model and merge
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        "/models/finetuned_qwen",
        device_map="auto",
        torch_dtype=torch.float16
    )
    merged_model = merged_model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained("/models/merged_finetuned_qwen")
    tokenizer.save_pretrained("/models/merged_finetuned_qwen")

    print("\n" + "="*80)
    print("SFT Phase Complete. Starting ART (GRPO) Reinforcement Learning Phase...")
    print("="*80 + "\n")

    # =============================================================================
    # PHASE 2: ART (GRPO) Reinforcement Learning Training
    # =============================================================================

    import os
    from art import GRPOTrainer
    import sys
    sys.path.insert(0, "/")  # Add root to path to import reward
    from reward import compute_reward

    # Load the original dataset again for GRPO (need prompts and ground truth)
    raw_dataset = load_dataset("virattt/financial-qa-10K", split="train")
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(1000))

    # Prepare dataset for GRPO: extract prompts and ground truth
    def prepare_grpo_dataset(example):
        # Create prompt in the same format as during SFT
        messages = [
            {"role": "system", "content": "You are a financial expert. Provide a concise answer to the question based on the given context."},
            {"role": "user", "content": f"Context: {example['context']}\n\nQuestion: {example['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "ground_truth": example['answer']
        }

    grpo_dataset = raw_dataset.map(prepare_grpo_dataset)

    # Create a reward function that uses the dataset
    def create_dataset_reward_fn(dataset):
        """Create a reward function that can look up ground truth from the dataset."""
        # Create a mapping from prompt to ground truth
        prompt_to_gt = {}
        for item in dataset:
            prompt_to_gt[item['prompt']] = item['ground_truth']
        
        def dataset_reward_fn(prompts, generations, **kwargs):
            rewards = []
            for prompt, generation in zip(prompts, generations):
                ground_truth = prompt_to_gt.get(prompt, "")
                reward = compute_reward(prompt, generation, ground_truth)
                rewards.append(reward)
            return rewards
        
        return dataset_reward_fn

    # Use the dataset-aware reward function
    reward_fn = create_dataset_reward_fn(grpo_dataset)

    # Load model for GRPO (use merged SFT model)
    print("Loading merged model for GRPO training...")
    grpo_model = merged_model

    print("Initializing GRPOTrainer...")
    rl_trainer = GRPOTrainer(
        model=grpo_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        rollout_batch_size=8,
        learning_rate=5e-6,
    )

    print("Starting GRPO training...")
    # GRPO training: pass dataset with prompts
    rl_trainer.train(grpo_dataset)

    print("Saving ART (GRPO) model...")
    # Ensure models directory exists
    os.makedirs("/models/qwen-4b-art", exist_ok=True)
    rl_trainer.save("/models/qwen-4b-art")

    print("\n" + "="*80)
    print("ART (GRPO) Training Complete!")
    print("Model saved to: /models/qwen-4b-art")
    print("="*80 + "\n")

    return "Fine-tuning (SFT + GRPO) completed and models saved."

@app.function(
    image=image,
    gpu=gpu.A100(),
    volumes={"/models": volume},
    timeout=1800
)
def run_inference(context: str, question: str, use_art: bool = False):
    import dspy
    from transformers import AutoTokenizer
    import os

    # Select model path based on use_art flag
    if use_art:
        model_path = "/models/qwen-4b-art"
        print(f"Loading ART (GRPO) model from: {model_path}")
    else:
        model_path = "/models/merged_finetuned_qwen"
        print(f"Loading SFT model from: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        return f"Error: Model not found at {model_path}. Please run finetune_model first."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set up DSPy with local fine-tuned model
    lm = dspy.HFModel(model=model_path, tokenizer=tokenizer, max_tokens=512, temperature=0.0)
    dspy.settings.configure(lm=lm)

    # Define a simple DSPy module for Financial QA with Chain-of-Thought
    class FinancialQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought("context: str, question: str -> answer: str")

        def forward(self, context, question):
            prediction = self.generate_answer(context=context, question=question)
            return prediction.answer

    # Run inference
    qa = FinancialQA()
    answer = qa(context=context, question=question)

    return answer

# To run locally or deploy
if __name__ == "__main__":
    # For local testing, but Modal functions run on cloud
    print("Run training with: modal run modal_app.py::finetune_model")
    print("Run inference (SFT): modal run modal_app.py::run_inference --context 'your context' --question 'your question'")
    print("Run inference (ART): modal run modal_app.py::run_inference --context 'your context' --question 'your question' --use-art")