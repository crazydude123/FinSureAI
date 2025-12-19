"""
Reward Function for Financial Q&A Fine-Tuning

This module provides the reward calculation logic for GRPO (Group Relative Policy Optimization)
training. It evaluates model responses based on:
- Semantic similarity to ground truth
- Key financial term matching
- Response length appropriateness
- Hallucination detection

Used by: openpipe/finetune_job.py
"""

from sentence_transformers import SentenceTransformer
import re

# Load semantic similarity model (cached after first load)
_similarity_model = None


def get_similarity_model():
    """Lazy load the sentence transformer model."""
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _similarity_model


def compute_reward(prompt, generated, ground_truth):
    """
    Compute reward for a generated answer given a prompt and ground truth.
    
    This function evaluates the quality of a model's response by combining multiple metrics:
    1. Semantic similarity (how close the meaning is)
    2. Key token matching (financial terms, numbers)
    3. Length penalty (too short or too long is bad)
    4. Hallucination penalty (making up information)
    
    Args:
        prompt (str): The input question/prompt
        generated (str): The model's generated answer
        ground_truth (str): The correct/expected answer
    
    Returns:
        float: Reward score between -1.0 and 1.0
               - 1.0 = Perfect answer
               - 0.0 = Mediocre answer
               - -1.0 = Terrible answer
    
    Example:
        >>> compute_reward(
        ...     "What was Apple's revenue in Q1?",
        ...     "Apple's Q1 revenue was $123.9 billion",
        ...     "Apple reported Q1 revenue of $123.9B"
        ... )
        0.85  # High reward for accurate, relevant answer
    """
    if not generated or not ground_truth:
        return -1.0
    
    # 1. Semantic Similarity (0.0 to 1.0)
    # Measures how similar the meaning is, regardless of exact wording
    model = get_similarity_model()
    embeddings = model.encode([generated, ground_truth])
    similarity = float(embeddings[0] @ embeddings[1])
    
    # 2. Key Token Matching
    # Extract important words (financial terms, numbers, entities)
    def extract_key_tokens(text):
        """Extract numbers, financial terms, and important words."""
        # Numbers (including percentages, currency)
        numbers = set(re.findall(r'\d+\.?\d*%?', text.lower()))
        # Financial keywords
        keywords = set(re.findall(r'\b(?:revenue|profit|loss|billion|million|quarter|q\d|fy\d+)\b', text.lower()))
        return numbers | keywords
    
    gen_tokens = extract_key_tokens(generated)
    truth_tokens = extract_key_tokens(ground_truth)
    
    if truth_tokens:
        token_overlap = len(gen_tokens & truth_tokens) / len(truth_tokens)
    else:
        token_overlap = 0.0
    
    # 3. Length Penalty
    # Penalize responses that are too short or too long
    gen_len = len(generated.split())
    truth_len = len(ground_truth.split())
    
    if truth_len > 0:
        length_ratio = gen_len / truth_len
        # Ideal ratio is 0.8 to 1.5 (slightly shorter to slightly longer)
        if 0.8 <= length_ratio <= 1.5:
            length_penalty = 0.0
        elif length_ratio < 0.5:
            length_penalty = -0.3  # Way too short
        elif length_ratio > 3.0:
            length_penalty = -0.3  # Way too long
        else:
            length_penalty = -0.1  # Slightly off
    else:
        length_penalty = 0.0
    
    # 4. Hallucination Detection
    # Penalize if generated text contains numbers not in ground truth
    gen_numbers = set(re.findall(r'\d+\.?\d*', generated))
    truth_numbers = set(re.findall(r'\d+\.?\d*', ground_truth))
    
    if gen_numbers and truth_numbers:
        # Check if generated numbers are wildly different
        hallucinated_numbers = gen_numbers - truth_numbers
        if hallucinated_numbers:
            hallucination_penalty = -0.2
        else:
            hallucination_penalty = 0.0
    else:
        hallucination_penalty = 0.0
    
    # 5. Repetition Penalty
    # Penalize if the response repeats itself
    words = generated.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.6:  # More than 40% repetition
            repetition_penalty = -0.2
        else:
            repetition_penalty = 0.0
    else:
        repetition_penalty = 0.0
    
    # Combine all components
    # Weights: similarity (60%), token overlap (30%), penalties (10%)
    reward = (
        0.6 * similarity +
        0.3 * token_overlap +
        length_penalty +
        hallucination_penalty +
        repetition_penalty
    )
    
    # Ensure reward is bounded between -1.0 and 1.0
    reward = max(-1.0, min(1.0, reward))
    
    return reward


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "prompt": "What was Apple's Q1 2024 revenue?",
            "generated": "Apple reported Q1 2024 revenue of $119.58 billion",
            "ground_truth": "Apple's Q1 2024 revenue was $119.58 billion",
            "expected_reward": 0.9  # Should be high
        },
        {
            "prompt": "What was Apple's Q1 2024 revenue?",
            "generated": "I don't know",
            "ground_truth": "Apple's Q1 2024 revenue was $119.58 billion",
            "expected_reward": -0.5  # Should be low
        },
        {
            "prompt": "What was Apple's Q1 2024 revenue?",
            "generated": "Apple made $999 billion in Q1",  # Wrong number
            "ground_truth": "Apple's Q1 2024 revenue was $119.58 billion",
            "expected_reward": 0.0  # Should be medium-low (hallucination)
        },
    ]
    
    print("Testing reward function:")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        reward = compute_reward(
            test["prompt"],
            test["generated"],
            test["ground_truth"]
        )
        print(f"\nTest {i}:")
        print(f"Generated: {test['generated']}")
        print(f"Ground Truth: {test['ground_truth']}")
        print(f"Reward: {reward:.3f} (expected ~{test['expected_reward']})")
