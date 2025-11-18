"""
Reward function for GRPO (Group Relative Policy Optimization) training.
Uses sentence-transformers for semantic similarity and heuristic checks.
"""

from sentence_transformers import SentenceTransformer
import re

# Initialize sentence transformer model (lazy loading)
_model = None

def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def compute_reward(prompt, generated, ground_truth):
    """
    Compute reward for a generated answer given a prompt and ground truth.
    
    Args:
        prompt: The input prompt/question
        generated: The generated answer from the model
        ground_truth: The ground truth answer
    
    Returns:
        float: Reward score (typically between -1 and 1)
    """
    if not generated or not ground_truth:
        return -0.5
    
    model = _get_model()
    
    # Base similarity score using sentence transformers
    embeddings = model.encode([generated, ground_truth], convert_to_tensor=True)
    similarity = float((embeddings[0] @ embeddings[1]) / (embeddings[0].norm() * embeddings[1].norm()))
    
    reward = similarity
    
    # Extract key tokens from ground truth (non-stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
    gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower())) - stop_words
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    
    # Bonus: key tokens from ground truth appear in generated
    key_matches = len(gt_words & gen_words)
    if key_matches > 0:
        reward += 0.1 * min(key_matches / max(len(gt_words), 1), 1.0)
    
    # Penalty: generated answer is much longer than ground truth
    len_ratio = len(generated) / max(len(ground_truth), 1)
    if len_ratio > 2.0:  # Generated is more than 2x longer
        reward -= 0.2
    
    # Penalty: repetition detection (simple heuristic)
    sentences = re.split(r'[.!?]+', generated)
    if len(sentences) > 1:
        # Check for repeated sentences
        unique_sentences = set(s.strip().lower() for s in sentences if len(s.strip()) > 10)
        if len(unique_sentences) < len(sentences) * 0.7:  # More than 30% repetition
            reward -= 0.2
    
    # Penalty: off-topic or hallucinated facts (simple heuristic)
    # Check if generated answer contains common hallucination indicators
    hallucination_indicators = [
        'i do not have access',
        'i cannot provide',
        'i am unable to',
        'as an ai',
        'i apologize',
    ]
    generated_lower = generated.lower()
    if any(indicator in generated_lower for indicator in hallucination_indicators):
        reward -= 0.2
    
    # Ensure reward is bounded
    reward = max(-1.0, min(1.0, reward))
    
    return float(reward)

