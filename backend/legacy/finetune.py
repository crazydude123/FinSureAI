import torch
import os
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
    output_dir="./finetuned_qwen",
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
trainer.save_model("./finetuned_qwen")

# Load PEFT model and merge
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    "./finetuned_qwen",
    device_map="auto",
    torch_dtype=torch.float16
)
merged_model = merged_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_finetuned_qwen")
tokenizer.save_pretrained("./merged_finetuned_qwen")

print("\n" + "="*80)
print("SFT Phase Complete. Starting ART (GRPO) Reinforcement Learning Phase...")
print("="*80 + "\n")

# =============================================================================
# PHASE 2: ART (GRPO) Reinforcement Learning Training
# =============================================================================

from art import GRPOTrainer
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

# Create a reward function wrapper that matches GRPO's expected signature
def grpo_reward_fn(prompts, generations, **kwargs):
    """
    Wrapper for compute_reward that works with GRPO's batch format.
    GRPO may pass additional kwargs, so we handle that.
    """
    rewards = []
    for prompt, generation in zip(prompts, generations):
        # Extract ground truth from dataset if available
        # For now, we'll use a simplified version that uses prompt context
        # In a real scenario, you'd match prompts to ground truth from the dataset
        ground_truth = kwargs.get('ground_truth', '')
        if not ground_truth:
            # Try to find ground truth from dataset
            # This is a simplified approach - in production, you'd maintain a mapping
            ground_truth = ""  # Will be handled in the training loop
        
        reward = compute_reward(prompt, generation, ground_truth)
        rewards.append(reward)
    return rewards

# Alternative: Create a proper reward function that uses the dataset
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

# Load model for GRPO (use merged SFT model, may need to load without quantization)
print("Loading merged model for GRPO training...")
# GRPO typically works better with full precision models, but we'll try with the merged model
# If needed, you can load without quantization:
# grpo_model = AutoModelForCausalLM.from_pretrained(
#     "./merged_finetuned_qwen",
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )

# For now, use the merged model we already have
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
# GRPOTrainer expects a dataset with 'prompt' field
rl_trainer.train(grpo_dataset)

print("Saving ART (GRPO) model...")
# Ensure models directory exists
os.makedirs("./models", exist_ok=True)
rl_trainer.save("./models/qwen-4b-art")

print("\n" + "="*80)
print("ART (GRPO) Training Complete!")
print("Model saved to: ./models/qwen-4b-art")
print("="*80 + "\n")