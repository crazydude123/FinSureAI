import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from reward import compute_reward
import evaluate
from tqdm import tqdm
import json
import os

def load_model(model_path):
    """Load the model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def prepare_test_dataset(dataset_name="virattt/financial-qa-10K", test_size=0.2, seed=42):
    """Load dataset and split into test set."""
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed)
    test_dataset = dataset.select(range(int(len(dataset) * test_size)))
    return test_dataset

def generate_answer(model, tokenizer, context, question, max_tokens=512, temperature=0.0):
    """Generate answer for a given context and question."""
    messages = [
        {"role": "system", "content": "You are a financial expert. Provide a concise answer to the question based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def compute_metrics(predictions, references):
    """Compute evaluation metrics."""
    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    # Semantic similarity using reward function
    semantic_scores = []
    for pred, ref in zip(predictions, references):
        # Create a dummy prompt for reward function
        prompt = "Financial QA"
        score = compute_reward(prompt, pred, ref)
        semantic_scores.append(score)

    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "semantic_similarity": avg_semantic
    }

def benchmark(model_path, test_size=0.2, num_samples=None, output_file="benchmark_results.json"):
    """Run benchmarking."""
    # Load model
    model, tokenizer = load_model(model_path)

    # Load test dataset
    test_dataset = prepare_test_dataset(test_size=test_size)
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

    print(f"Testing on {len(test_dataset)} samples")

    predictions = []
    references = []
    results = []

    for sample in tqdm(test_dataset, desc="Generating answers"):
        context = sample['context']
        question = sample['question']
        ground_truth = sample['answer']

        generated = generate_answer(model, tokenizer, context, question)
        predictions.append(generated)
        references.append(ground_truth)

        results.append({
            "context": context,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": generated
        })

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Save results
    output = {
        "metrics": metrics,
        "num_samples": len(test_dataset),
        "model_path": model_path,
        "results": results
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("Benchmarking complete!")
    print(f"Results saved to: {output_file}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Qwen model on financial QA dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (local or Hugging Face repo)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of dataset to use for testing")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to test (overrides test_size)")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Output file for results")

    args = parser.parse_args()
    benchmark(args.model_path, args.test_size, args.num_samples, args.output_file)