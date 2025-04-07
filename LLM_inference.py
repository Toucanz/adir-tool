from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import re
from LLM_trainer import add_tags, add_instructions, CoT_instructions

############# Control Panel: ##############
modelLoad = True
runInference = True
###########################################
LoRA_wrap = True

###########################################
################ Load Model ###############
###########################################

if modelLoad:
    # Paths to saved models
    base_model_name = "google/flan-t5-base"
    if LoRA_wrap: model_path = "./flan-t5-lora-adapter-dialougeSum_CT1"
    else: model_path = "./flan-t5-finetuned"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if LoRA_wrap:
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        # Load LoRA adapter weights into the base model
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load the fine-tuned model directly
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Put the model in evaluation mode
    model.eval()



###########################################
############# Prepare Inputs ##############
###########################################

new_inputs = [
    "He would always look at me.",
    "She often avoided eye contact.",
    "She didn't seem to ever look at me.",
    "I don't think she looked at me when I entered the room.",
    "He sometimes looked at me when I entered the room.",
    "He would glance at me, then look away quickly.",
    "I don't think she ever looks at me.",
    "At that age, he did look at me sometimes, but not often. If we were playing together, I could usually catch his eye for a second or two, but then he’d look away. When I walked into the room, he didn’t really watch me like other kids might. He seemed more focused on whatever he was doing. With others, it was about the same, he didn’t seek out eye contact much, even with family members.",
    "Oh yes, she made eye contact all the time. When we talked, she looked at my face, and if I walked into the room, she’d turn to look at me. If we were playing a game, she’d check my expression a lot. She did that with other people, too—not just me.",
    "Not really. It was hard to get his attention that way. I’d try to get him to look at me, but his eyes would kind of dart around instead of meeting mine. If I walked into the room, he didn’t seem to notice unless I spoke. With other people, he mostly avoided eye contact.",
    "We like to swim together and eat food. She would always look at me when I entered the room.",
    "Good question, i'm not honestly sure! But I think he does look at strangers sometimes.",
    "They don't often look at strangers. But I do need to leave soon, I have a meeting to attend.",
    "When we go to the park, they always look, but don't look very long.",
    "We love playing tennis together, and he makes great eye contact when we play. But when talking, he doesn't look at me much.",
    "I don't think she looks at me when we eat as a family, but she does look at her siblings when they talk. I think she likes me.",
    "Jim gives me strange looks, I can't tell if he really notices me or not. We would be talking, and his vision is on me, but he seems distant",
    "I have no idea if she looks at me or not. I am a bad parent",
    "When I take her to the store, she's always looking at the candy.  When I call her name, she comes to me, but she doesn't look at me.",
]

prepared_inputs = list(map(add_instructions, new_inputs))

#print(prepared_inputs)

###########################################
############## Run Inference ##############
###########################################

if runInference:
        # Tokenize and move to model's device
    inputs = tokenizer(
        prepared_inputs, 
        return_tensors="pt", 
        padding=True, 
        truncation=True).to(model.device)

    # Generate outputs
    outputs = model.generate(
        **inputs,
        min_length=4,
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