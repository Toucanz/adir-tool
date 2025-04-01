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
    "Well, I never really thought about it like that. I mean, he was always kind of in his own world, you know? Like, he’d be playing with his toys, and I’d try to talk to him, and sometimes he’d glance up, but not always. I don’t know if it was that he didn’t want to look or if he was just too focused on whatever he was doing. He was really into his trains back then, so if he was lining them up, forget it—nothing else existed. Actually, thinking about it, he did not really pay attention to what anyone else if he played with his toys. No, he always seemed to be in his own world at that age, not paying attention to what anyone else is saying",
    "Oh yes, she made eye contact all the time. She would look atme for reassurance if she was adventorous, or if she did something cheeky. Other parents did comment on how sociable she was for a 5 year old"
]



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