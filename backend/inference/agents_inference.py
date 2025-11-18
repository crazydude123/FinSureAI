import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

def generate(model, tokenizer, messages, max_tokens=1024, temperature=0.7, seed=None):
    if seed is not None:
        set_seed(seed)
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Self-critical inference for proof verification using agentic loop")
    parser.add_argument("--model_path", type=str, default="./merged_finetuned_qwen", help="Path to the finetuned model")
    parser.add_argument("--proof", type=str, default="Proof that 1=2:\nAssume a = b.\nThen a^2 = ab.\nSubtract b^2: a^2 - b^2 = ab - b^2.\n(a - b)(a + b) = b(a - b).\nDivide both sides by (a - b): a + b = b.\nSubstitute a = 1: 1 + b = b => 1 = 0.", help="The proof or argument to verify")
    parser.add_argument("--question", type=str, default="Is this proof correct? What is the logical inconsistency in this argument?", help="Question about the proof")
    parser.add_argument("--num_iters", type=int, default=3, help="Number of self-critique iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if None)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Generate random seed if not provided
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"Using random seed: {seed}")
    set_seed(seed)
    
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initial verification messages
    system = "You are an AI scientist specialized in verifying mathematical proofs and logical arguments. Be precise, thorough, and identify any inconsistencies, unproven assumptions, invalid steps, or errors."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Proof/Argument:\n{args.proof}\n\nQuestion: {args.question}\n\nProvide a detailed analysis."}
    ]
    
    print("\n" + "="*80)
    print("INITIAL VERIFICATION:")
    print("="*80)
    initial_analysis = generate(model, tokenizer, messages, max_tokens=args.max_tokens, temperature=args.temperature * 0.8, seed=seed)
    print(initial_analysis)
    print()
    
    current_analysis = initial_analysis
    conversation = messages + [{"role": "assistant", "content": initial_analysis}]
    
    stopped_early = False
    for i in range(args.num_iters):
        subseed = int((seed + (i + 1) * 12345) % (2**31))
        critique_seed = int((subseed + 54321) % (2**31))
        
        # Critique the current analysis
        critique_system = "You are a harsh, meticulous self-critic for proof verifications. Identify ANY logical holes, inconsistencies, overlooked assumptions, invalid inferences, missed edge cases, or ways to improve clarity/accuracy. Be thorough and specific."
        critique_messages = [
            {"role": "system", "content": critique_system},
            {"role": "user", "content": f"""Previous analysis:
{current_analysis}

Original Proof/Argument:
{args.proof}

Original Question:
{args.question}

Provide a detailed critique, pointing out weaknesses and suggesting revisions:"""}
        ]
        
        print(f"\nIteration {i+1}: CRITIQUE (seed={critique_seed}):")
        critique = generate(model, tokenizer, critique_messages, max_tokens=args.max_tokens//2, temperature=args.temperature * 1.1, seed=critique_seed)
        print(critique)
        print()
        
        # Check if critique indicates no major issues
        critique_lower = critique.lower()
        if any(phrase in critique_lower for phrase in ["no major issues", "solid", "correct overall", "no inconsistencies", "valid proof"]):
            print("Critique indicates no major issues. Stopping early.")
            stopped_early = True
            break
        
        # Reprompt for revision
        revise_messages = conversation + [
            {"role": "user", "content": f"Critique of your analysis:\n{critique}\n\nRevise your analysis to address all points in the critique. Improve accuracy, fill gaps, and ensure logical soundness."}
        ]
        
        revise_seed = int((critique_seed + 11111) % (2**31))
        print(f"Iteration {i+1}: REVISED ANALYSIS (seed={revise_seed}):")
        revised_analysis = generate(model, tokenizer, revise_messages, max_tokens=args.max_tokens, temperature=args.temperature * 0.6, seed=revise_seed)
        print(revised_analysis)
        print()
        
        current_analysis = revised_analysis
        conversation.append({"role": "assistant", "content": revised_analysis})
    
    print("\n" + "="*80)
    print("FINAL SELF-VERIFIED ANALYSIS:")
    print("="*80)
    print(current_analysis)
    print("\nIterations completed:", i+1 if stopped_early else args.num_iters)
    print("Seed used:", seed)

if __name__ == "__main__":
    main()