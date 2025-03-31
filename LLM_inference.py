from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import re

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
    "He usually avoided eye contact. I could call his name or try to talk to him, and he’d just look away or focus on a toy. I don’t remember him looking at me when I came into the room. He didn’t seem interested in people’s faces.",
    "She was always looked at me when we were talking. If I walked into a room, she’d look right at me. During play, she’d check my face to see if I was smiling or paying attention. She also did that with other people, not just me.",
    "At that age, he would sometimes look at me when I talked to him, but it didn’t last long. If I tried to get his attention, he'd glance up, but then go right back to what he was doing. With strangers or even family, he rarely made eye contact."
]



def add_tags(input_text):
    words_to_replace = ("she", "he", "his", "her", "she's", "he's", "he'd", "she'd", "she'll", "he'll")
    input_text = input_text.replace("’", "'")   # Fix stupid apostrophes

    ### Tokenizing the text while keeping punctuation
    ### This regex will match words and punctuation
    pattern = r"[^\s.,!?;:]+|[.,!?;:\s]"
    input_tokens = re.findall(pattern, input_text)

    #### Rebuild the input and output texts
    input_text_out = ''.join([f"[CHILD]{word}[/CHILD]" if word.lower() in words_to_replace else word for word in input_tokens])

    return input_text_out


def add_instructions(input_text):
    instruction = "TASK: Based on the parent's statement, describe the child's eye contact patterns in a clinical tone"
    response = f"PARENT RESPONSE: {input_text}"
    full_input = f"{instruction}\n\n{response}"
    input_text = full_input
    return input_text

prepared_inputs = list(map(add_instructions, list(map(add_tags, new_inputs))))

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
        min_length=16,
        max_length=256,
        num_beams=8,           # Beam search for better quality (optional)
        early_stopping=True
    )

    # Decode predictions
    summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Print the results
    for idx, summary in enumerate(summaries):
        print(f"Input: {new_inputs[idx]}\n\n")
        print(f"Predicted Summary: {summary}\n\n")