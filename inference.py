import dspy
import argparse
from transformers import AutoTokenizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with FinSure AI model")
    parser.add_argument(
        "--use_art",
        action="store_true",
        help="Use ART (GRPO) fine-tuned model instead of SFT model"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="Since our original focus on PC graphics, we have expanded to several large and important computationally intensive fields.",
        help="Context for the question"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What area did NVIDIA initially focus on before expanding?",
        help="Question to answer"
    )
    args = parser.parse_args()

    # Select model path based on flag
    if args.use_art:
        model_path = "./models/qwen-4b-art"
        print(f"Loading ART (GRPO) model from: {model_path}")
    else:
        model_path = "./merged_finetuned_qwen"
        print(f"Loading SFT model from: {model_path}")

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
    answer = qa(context=args.context, question=args.question)
    
    print(f"\nContext: {args.context}")
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()