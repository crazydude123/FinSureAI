"""OpenPipe ART fine-tuning job for FinSureAI.

This script mirrors the structure of modal/finetune_job.py but uses
OpenPipe ART framework instead of Modal for GPU-accelerated training.
"""

import os
import sys
import asyncio
import random
from typing import Optional
from dotenv import load_dotenv

import art
from art.serverless.backend import ServerlessBackend
# Alternative for local GPU training:
# from art.local.backend import LocalBackend

import wandb
import weave
from openai import AsyncOpenAI
from datasets import load_dataset
from pydantic import BaseModel
import requests

# Import reward function from modal directory (reuse existing logic)
from reward import compute_reward

load_dotenv()

# Verify W&B API key
if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required. Set it in .env file.")

# Supported models (same as modal/finetune_job.py)
SUPPORTED_MODELS = (
    'OpenPipe/Qwen3-14B-Instruct',
    'Qwen/Qwen3-4B-Instruct-2507',
    'meta-llama/Llama-3.1-70B-Instruct'
)

# Dataset configuration
DATASET_NAME = 'virattt/financial-qa-10K'
TRAIN_SIZE = 1000  # Number of examples to use for training


class FinancialScenario(BaseModel):
    """Scenario for financial Q&A rollout."""
    step: int
    context: str
    question: str
    answer: str
    id: str = ""


def load_financial_dataset(dataset_url: Optional[str] = None):
    """
    Load financial dataset from Hugging Face or URL.
    
    Args:
        dataset_url: Optional URL or Hugging Face dataset name
                    Defaults to virattt/financial-qa-10K
    
    Returns:
        Dataset object
    """
    if dataset_url and dataset_url.startswith('http'):
        # Download from URL (e.g., Supabase Storage)
        print(f"📥 Downloading dataset from {dataset_url}")
        response = requests.get(dataset_url, timeout=120)
        response.raise_for_status()
        
        with open('/tmp/dataset.jsonl', 'wb') as f:
            f.write(response.content)
        
        dataset = load_dataset('json', data_files={'train': '/tmp/dataset.jsonl'})['train']
    else:
        # Load from Hugging Face
        dataset_name = dataset_url or DATASET_NAME
        print(f"📥 Loading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')
    
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    return dataset


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def financial_rollout(model: art.Model, scenario: FinancialScenario) -> art.Trajectory:
    """
    Rollout function for financial Q&A.
    
    This function:
    1. Creates a trajectory with system and user messages
    2. Gets model response via OpenAI-compatible API
    3. Calculates reward using the reward function from modal/reward.py
    4. Returns trajectory with reward
    """
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are a financial expert. Provide accurate, concise answers based on the given context. Focus on factual information from the context."
            },
            {
                "role": "user",
                "content": f"Context: {scenario.context}\n\nQuestion: {scenario.question}"
            }
        ],
        metadata={
            "scenario_id": scenario.id,
            "step": scenario.step,
            "dataset": DATASET_NAME,
        },
        reward=0,
    )
    
    try:
        # Get model response
        messages = trajectory.messages()
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=512,
            messages=messages,
            model=model.get_inference_name(),
            temperature=0.7,
        )
        
        # Add choice to trajectory
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str), "Response content must be string"
        trajectory.messages_and_choices.append(choice)
        
        # Calculate reward using existing reward function
        trajectory.reward = compute_reward(
            scenario.question,
            content,
            scenario.answer
        )
        
        # Add metrics
        trajectory.metrics["response_length"] = len(content)
        trajectory.metrics["ground_truth_length"] = len(scenario.answer)
        trajectory.metrics["context_length"] = len(scenario.context)
        
    except Exception as e:
        print(f"❌ Error in rollout: {e}")
        trajectory.reward = -1.0
        raise e
    
    return trajectory


async def finetune_job(
    dataset_url: Optional[str] = None,
    model_name: str = 'OpenPipe/Qwen3-14B-Instruct',
    user_id: str = 'default',
    num_steps: int = 20,
    rollouts_per_step: int = 18,
    learning_rate: float = 1e-5
):
    """
    Main fine-tuning job function.
    
    Args:
        dataset_url: URL or Hugging Face dataset name
        model_name: Base model to fine-tune
        user_id: User identifier for tracking
        num_steps: Number of training steps
        rollouts_per_step: Number of parallel rollouts per step
        learning_rate: Learning rate for training
    
    Returns:
        dict with model info and inference URL
    """
    
    # Validate model
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model. Choose from: {SUPPORTED_MODELS}")
    
    # Load dataset
    dataset = load_financial_dataset(dataset_url)
    
    # Take subset for training
    train_dataset = dataset.select(range(min(TRAIN_SIZE, len(dataset))))
    print(f"📊 Training on {len(train_dataset)} examples")
    
    # Create trainable model
    print(f"🤖 Creating model: {model_name}")
    model = art.TrainableModel(
        name=f"finsureai-{user_id}",
        project="finsureai",
        base_model=model_name,
    )
    
    # Initialize backend
    # Option 1: ServerlessBackend (W&B Training - autoscaling GPUs)
    backend = ServerlessBackend()
    
    # Option 2: LocalBackend (for local GPU training)
    # backend = LocalBackend(
    #     gpu_count=1,
    #     gpu_type="A100",
    # )
    
    # Register model with backend
    await model.register(backend)
    
    print(f"✅ Model registered: {model.name}")
    print(f"📍 Project: {model.project}")
    print(f"🔗 W&B URL: https://wandb.ai/{model.project}")
    
    # Initialize Weave for logging
    weave.init(
        model.project,
        settings={"print_call_link": False},
    )
    
    # Training loop
    print(f"\n🚀 Starting training:")
    print(f"   Steps: {num_steps}")
    print(f"   Rollouts per step: {rollouts_per_step}")
    print(f"   Learning rate: {learning_rate}")
    
    for step in range(await model.get_step(), num_steps):
        print(f"\n📊 Step {step}/{num_steps}")
        
        # Create scenarios from dataset
        scenarios = []
        for i in range(rollouts_per_step):
            # Sample from dataset (with replacement)
            idx = random.randint(0, len(train_dataset) - 1)
            example = train_dataset[idx]
            
            scenarios.append(FinancialScenario(
                step=step,
                context=example.get('context', ''),
                question=example.get('question', ''),
                answer=example.get('answer', example.get('response', '')),
                id=str(example.get('id', idx))
            ))
        
        # Gather trajectories
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(financial_rollout(model, s) for s in scenarios)
                for _ in range(1)
            ),
            pbar_desc=f"Step {step}",
            max_exceptions=rollouts_per_step,
        )
        
        # Delete old checkpoints (keep only best and recent)
        await model.delete_checkpoints('train/reward')
        
        # Train model
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=learning_rate),
        )
        
        print(f"✅ Step {step} complete")
    
    print("\n🎉 Training complete!")
    print(f"🔗 View results: https://wandb.ai/{model.project}")
    
    # Return inference info
    return {
        'model_name': model.get_inference_name(),
        'inference_url': model.inference_base_url,
        'wandb_url': f"https://wandb.ai/{model.project}",
        'user_id': user_id,
    }


if __name__ == "__main__":
    # Parse command line arguments
    dataset_url = sys.argv[1] if len(sys.argv) > 1 else None
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'OpenPipe/Qwen3-14B-Instruct'
    user_id = sys.argv[3] if len(sys.argv) > 3 else 'default'
    
    # Run training
    result = asyncio.run(finetune_job(dataset_url, model_name, user_id))
    
    print(f"\n✅ Training complete!")
    print(f"Model: {result['model_name']}")
    print(f"Inference URL: {result['inference_url']}")
    print(f"W&B URL: {result['wandb_url']}")
