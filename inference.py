import dspy
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./merged_finetuned_qwen")

# Set up DSPy with local fine-tuned model
lm = dspy.HFModel(model="./merged_finetuned_qwen", tokenizer=tokenizer, max_tokens=512, temperature=0.0)
dspy.settings.configure(lm=lm)

# Define a simple DSPy module for Financial QA with Chain-of-Thought
class FinancialQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("context: str, question: str -> answer: str")

    def forward(self, context, question):
        prediction = self.generate_answer(context=context, question=question)
        return prediction.answer

# Example usage
qa = FinancialQA()

# Test with a sample (replace with your own)
test_context = "Since our original focus on PC graphics, we have expanded to several large and important computationally intensive fields."
test_question = "What area did NVIDIA initially focus on before expanding?"
answer = qa(context=test_context, question=test_question)
print(f"Answer: {answer}")