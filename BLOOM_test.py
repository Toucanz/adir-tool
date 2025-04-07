from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load BLOOM model and tokenizer
model_name = "bigscience/bloom-560m"  # You can try bloomz too
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt for generation
prompt = "The results of the dyslexia screening suggest that"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nInput prompt:  {prompt}\n\nPredicted Output:\n{generated_text}")