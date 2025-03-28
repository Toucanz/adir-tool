from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

############# Control Panel: ##############
modelLoad = True
runInference = True
###########################################


###########################################
################ Load Model ###############
###########################################

if modelLoad:
    # Paths to saved models
    base_model_name = "google/flan-t5-base"
    lora_adapter_dir = "./flan-t5-lora-adapter"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_dir)

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # Load LoRA adapter weights into the base model
    model = PeftModel.from_pretrained(base_model, lora_adapter_dir)

    # Put the model in evaluation mode
    model.eval()



###########################################
############# Prepare Inputs ##############
###########################################

new_inputs = [
    "At that age, he did look at me sometimes, but not often.",
    "Oh yes, she made eye contact all the time."
]

def add_instructions(example):
    instruction = "TASK: You are a clinial interviewer.  You just asked a parent to describe their child's eye contact behaviour.  Their response is below.  Write a objective summary of the child's eye contact behaviour."
    response = f"PARENT RESPONSE: {example}"
    full_input = f"{instruction}\n\n{response}"
    example = full_input
    return example

prepared_inputs = list(map(add_instructions, new_inputs))

print(prepared_inputs)

###########################################
############## Run Inference ##############
###########################################

if runInference:
        # Tokenize and move to model's device
    inputs = tokenizer(new_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate outputs
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,           # Beam search for better quality (optional)
        early_stopping=True
    )

    # Decode predictions
    summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Print the results
    for idx, summary in enumerate(summaries):
        print(f"Input: {new_inputs[idx]}")
        print(f"Predicted Summary: {summary}\n")