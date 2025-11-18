"""
Test script to verify dataset connectivity and test reward function.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from legacy.reward import compute_reward

print("=" * 80)
print("Testing Dataset Connection and Reward Function")
print("=" * 80)

# Test 1: Load dataset
print("\n[TEST 1] Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("virattt/financial-qa-10K", split="train")
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Features: {list(dataset.features.keys())}")
    
    # Show a sample
    sample = dataset[0]
    print(f"\n  Sample example:")
    print(f"  - Context: {sample.get('context', 'N/A')[:100]}...")
    print(f"  - Question: {sample.get('question', 'N/A')}")
    print(f"  - Answer: {sample.get('answer', 'N/A')[:100]}...")
    
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# Test 2: Test tokenizer and formatting
print("\n[TEST 2] Testing tokenizer and chat template formatting...")
try:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded successfully!")
    
    # Test formatting
    example = dataset[0]
    messages = [
        {"role": "system", "content": "You are a financial expert. Provide a concise answer to the question based on the given context."},
        {"role": "user", "content": f"Context: {example['context']}\n\nQuestion: {example['question']}"},
        {"role": "assistant", "content": example['answer']}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(f"✓ Chat template formatting works!")
    print(f"  - Formatted text length: {len(formatted_text)} chars")
    print(f"  - Preview: {formatted_text[:200]}...")
    
except Exception as e:
    print(f"✗ Error with tokenizer: {e}")
    exit(1)

# Test 3: Test reward function
print("\n[TEST 3] Testing reward function...")
try:
    # Test with sample data
    prompt = f"Context: {sample['context']}\n\nQuestion: {sample['question']}"
    ground_truth = sample['answer']
    
    # Test with correct answer
    reward_correct = compute_reward(prompt, ground_truth, ground_truth)
    print(f"✓ Reward function works!")
    print(f"  - Reward (correct answer): {reward_correct:.4f}")
    
    # Test with different answer
    test_answer = "This is a test answer that might not match."
    reward_test = compute_reward(prompt, test_answer, ground_truth)
    print(f"  - Reward (test answer): {reward_test:.4f}")
    
    # Test with empty answer
    reward_empty = compute_reward(prompt, "", ground_truth)
    print(f"  - Reward (empty answer): {reward_empty:.4f}")
    
except Exception as e:
    print(f"✗ Error with reward function: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test GRPO dataset preparation
print("\n[TEST 4] Testing GRPO dataset preparation...")
try:
    raw_dataset = dataset.shuffle(seed=42).select(range(10))  # Just 10 for testing
    
    def prepare_grpo_dataset(example):
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
    print(f"✓ GRPO dataset preparation works!")
    print(f"  - GRPO dataset size: {len(grpo_dataset)}")
    print(f"  - Sample prompt length: {len(grpo_dataset[0]['prompt'])} chars")
    print(f"  - Sample ground truth length: {len(grpo_dataset[0]['ground_truth'])} chars")
    
    # Test reward function with GRPO format
    def create_dataset_reward_fn(dataset):
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
    
    reward_fn = create_dataset_reward_fn(grpo_dataset)
    test_prompts = [grpo_dataset[0]['prompt']]
    test_generations = [grpo_dataset[0]['ground_truth']]
    test_rewards = reward_fn(test_prompts, test_generations)
    print(f"✓ Reward function wrapper works!")
    print(f"  - Test reward: {test_rewards[0]:.4f}")
    
except Exception as e:
    print(f"✗ Error with GRPO dataset preparation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Check dependencies
print("\n[TEST 5] Checking required dependencies...")
try:
    import art
    print(f"✓ art-openpipe installed: {art.__version__ if hasattr(art, '__version__') else 'version unknown'}")
except ImportError:
    print(f"⚠ art-openpipe not installed - install with: pip install art-openpipe>=0.1.7")

try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ sentence-transformers installed")
except ImportError:
    print(f"⚠ sentence-transformers not installed - install with: pip install sentence-transformers")

try:
    from trl import SFTTrainer
    print(f"✓ trl installed")
except ImportError:
    print(f"⚠ trl not installed - install with: pip install trl>=0.8.0")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
print("\n✓ Dataset connection: WORKING")
print("✓ Tokenizer: WORKING")
print("✓ Reward function: WORKING")
print("✓ GRPO dataset preparation: WORKING")
print("\nReady to run training with: python backend/legacy/finetune.py")

